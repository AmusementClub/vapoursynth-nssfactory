#include "nss/resources.hpp"
#include "nss/workspace.hpp"
#include "nss/cpu_lssc.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include <cstdio>
#include <array>
#include <barrier>
#include <condition_variable>
#include <set>
#include <thread>
#include <vector>

int main() {
    auto budget = std::make_shared<nss::ResourceBudget>(1<<20);
    {
        nss::ResourceScope scope(budget);
        nss::Workspace scratch;
        scratch.set_serial();
        float* first = nullptr;
        bool okay = true;
        std::barrier ready(9);
        std::mutex handoff;
        std::condition_variable cv;
        int turn = 0;
        std::set<std::thread::id> ids;
        std::array<std::thread, 8> workers;
        for (int worker = 0; worker < 8; ++worker) {
            workers[worker] = std::thread([&, worker] {
                ready.arrive_and_wait();
                std::unique_lock lock(handoff);
                cv.wait(lock, [&] { return turn == worker; });
                ids.insert(std::this_thread::get_id());
                auto* buffer = scratch.get(16384);
                if (!first) first = buffer;
                else if (buffer != first) okay = false;
                buffer[16383] = 3.f;
                ++turn;
                cv.notify_all();
            });
        }
        ready.arrive_and_wait();
        for (auto& worker : workers) worker.join();
        if (!okay || ids.size() != 8 || scratch.buffer_count() != 1) return 1;
        {
            nss::Workspace parallel;
            std::barrier start(9), written(9);
            std::array<float*, 8> pointers{};
            std::array<bool, 8> valid{};
            for (int worker = 0; worker < 8; ++worker) workers[worker] = std::thread([&, worker] {
                start.arrive_and_wait();
                pointers[worker] = parallel.get(16384);
                pointers[worker][16383] = static_cast<float>(worker);
                written.arrive_and_wait();
                valid[worker] = pointers[worker][16383] == static_cast<float>(worker);
            });
            start.arrive_and_wait(); written.arrive_and_wait();
            for (auto& worker : workers) worker.join();
            if (parallel.buffer_count() != 8 || std::set<float*>(pointers.begin(), pointers.end()).size() != 8)
                return 1;
            for (bool value : valid) if (!value) return 1;
        }
        auto chunk = nss::make_resource_account(nss::ResourceKind::Inflight);
        {
            nss::ResourceScope chunk_scope(budget, chunk);
            nss::ResourceVector<float> data(1024);
            chunk->retag(nss::ResourceKind::Cached);
            auto snap = budget->snapshot();
            if (snap.bytes[static_cast<int>(nss::ResourceKind::Cached)] != 4096) return 1;
            chunk->retag(nss::ResourceKind::Pinned);
            snap = budget->snapshot();
            if (snap.bytes[static_cast<int>(nss::ResourceKind::Cached)] ||
                snap.bytes[static_cast<int>(nss::ResourceKind::Pinned)] != 4096) return 1;
            bool rejected = false;
            try { data.resize(1<<20); } catch (const std::runtime_error&) { rejected = true; }
            if (!rejected || data.size() != 1024) return 1;
        }
        if (budget->snapshot().bytes[static_cast<int>(nss::ResourceKind::Pinned)]) return 1;
    }
    const auto snap = budget->snapshot();
    if (snap.owned) return 1;
    for (auto bytes : snap.bytes) if (bytes) return 1;
    for (int dimension : {100000, 2147483647}) {
        bool rejected = false;
        try { (void)nss::lssc_denoise_work_floats(dimension,dimension,8,1); }
        catch (const std::exception&) { rejected = true; }
        if (!rejected) return 1;
    }
    for (auto sizing : {nss::wnnm_shrink_work_floats, nss::twsc_pca_soft_work_floats,
                        nss::mcwnnm_filter_work_floats, nss::bm3d_filter_work_floats}) {
        bool rejected = false;
        try { (void)sizing(2147483647,2147483647); } catch (const std::exception&) { rejected = true; }
        if (!rejected) return 1;
    }
    bool rejected = false;
    try { (void)nss::ncsr_denoise_work_floats(2147483647,2147483647,8,8); }
    catch (const std::exception&) { rejected = true; }
    if (!rejected) return 1;
    {
        // Exercise a CPU-library allocation from a separately compiled TU.
        // All input/output arrays are valid even if accounting were broken.
        constexpr int count = 10000;
        std::vector<nss::MatchBatchItem> items(count, nss::MatchBatchItem{0,0,1,0,1});
        std::vector<nss::Match> matches(count);
        std::vector<int> counts(count);
        auto tiny = std::make_shared<nss::ResourceBudget>(4096);
        nss::ResourceScope scope(tiny);
        float input = .25f;
        bool denied = false;
        try { (void)nss::spatial_match_batch(&input,1,1,1,items.data(),count,matches.data(),1,counts.data()); }
        catch (const std::runtime_error&) { denied = true; }
        if (!denied || tiny->snapshot().owned) return 1;
    }
    std::puts("eight-worker serial handoff, pinned accounting, budget failure and teardown passed");
    return 0;
}
