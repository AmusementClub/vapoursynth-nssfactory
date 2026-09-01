#pragma once

#include "nss/cpu_api.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace nss {

inline constexpr int kSvdBatch8MaxM = 64;

// Internal CPU scheduling metadata. It is intentionally separate from the
// VapourSynth map contract: users cannot select a batch shape or a scheduler.
enum class GroupAlgorithm : std::uint8_t { BM3D, WNNM, TWSC, MCWNNM, NCSR, NLH, LSSC };

struct GroupKey {
    int m = 0;
    int k = 0;
    int channels = 1;
    GroupAlgorithm algorithm = GroupAlgorithm::BM3D;
    bool basic = false;
    bool residual = false;

    friend bool operator==(const GroupKey& a, const GroupKey& b) noexcept {
        return a.m == b.m && a.k == b.k && a.channels == b.channels && a.algorithm == b.algorithm &&
               a.basic == b.basic && a.residual == b.residual;
    }

    friend bool operator<(const GroupKey& a, const GroupKey& b) noexcept {
        if (a.m != b.m) {
            return a.m < b.m;
        }
        if (a.k != b.k) {
            return a.k < b.k;
        }
        if (a.channels != b.channels) {
            return a.channels < b.channels;
        }
        if (a.algorithm != b.algorithm) {
            return static_cast<std::uint8_t>(a.algorithm) < static_cast<std::uint8_t>(b.algorithm);
        }
        if (a.basic != b.basic) {
            return a.basic < b.basic;
        }
        return a.residual < b.residual;
    }
};

struct GroupJob {
    std::uint64_t ordinal = 0;
    int x = 0;
    int y = 0;
    int t = 0;
    GroupKey key{};
};

// A bounded reorder queue. Producers may submit completed jobs in any order;
// the consumer sees only the original raster ordinal order.
template <typename Result>
class OrderedCommitQueue {
public:
    explicit OrderedCommitQueue(std::uint64_t first_ordinal = 0, std::size_t window = 32)
        : next_ordinal_(first_ordinal), window_(std::max<std::size_t>(1, window)) {
        pending_.reserve(window_ + 1);
    }

    bool push(std::uint64_t ordinal, Result result) {
        if (ordinal < next_ordinal_) {
            return false;
        }
        // Keep the amount of speculative work bounded.  Callers must drain
        // the queue before submitting an ordinal outside this window.
        if (ordinal - next_ordinal_ > window_) {
            return false;
        }
        for (const auto& item : pending_) {
            if (item.ordinal == ordinal) {
                return false;
            }
        }
        pending_.push_back(Item{ordinal, std::move(result)});
        return true;
    }

    template <typename CommitFn>
    std::size_t drain(CommitFn&& commit) {
        std::size_t committed = 0;
        while (true) {
            auto it = std::find_if(pending_.begin(), pending_.end(),
                                   [&](const Item& item) { return item.ordinal == next_ordinal_; });
            if (it == pending_.end()) {
                break;
            }
            commit(it->result);
            pending_.erase(it);
            ++next_ordinal_;
            ++committed;
        }
        return committed;
    }

    template <typename CommitFn>
    std::size_t finish(CommitFn&& commit) {
        // Finish is deliberately ordered too.  If a producer forgot an
        // ordinal, leave the later results pending instead of silently
        // changing floating-point accumulation order.
        return drain(std::forward<CommitFn>(commit));
    }

    bool needs_drain() const noexcept { return pending_.size() > window_; }
    std::size_t pending() const noexcept { return pending_.size(); }
    std::uint64_t next_ordinal() const noexcept { return next_ordinal_; }
    std::size_t window() const noexcept { return window_; }
    bool complete() const noexcept { return pending_.empty(); }

private:
    struct Item {
        std::uint64_t ordinal;
        Result result;
    };
    std::uint64_t next_ordinal_;
    std::size_t window_;
    std::vector<Item> pending_;
};

inline void bucket_group_jobs(std::vector<GroupJob>& jobs) {
    std::stable_sort(jobs.begin(), jobs.end(), [](const GroupJob& a, const GroupJob& b) {
        if (a.key == b.key) {
            return a.ordinal < b.ordinal;
        }
        return a.key < b.key;
    });
}

struct MatchBatchItem {
    int bx = 0;
    int by = 0;
    int block = 8;
    int bm_range = 7;
    int group = 8;
};

// Results for item i are written at matches + i * match_stride and counts[i].
// The wrappers sort work by dimensions internally, then restore caller order.
int spatial_match_batch(const float* ref, int stride, int width, int height, const MatchBatchItem* items, int count,
                        Match* matches, int match_stride, int* counts);
int spatial_match_nch_batch(const float* const* refs, const int* strides, int nch, int width, int height,
                            const MatchBatchItem* items, int count, Match* matches, int match_stride, int* counts);

// Predictive variant used by temporal filters.  The frame pointer layout is
// the same as predictive_match: refs[t], strides[t].
int predictive_match_batch(const float* const* refs, const int* strides, int ntemp, int width, int height, int t0,
                           const SearchConfig& cfg, const MatchBatchItem* items, int count, Match* matches,
                           int match_stride, int* counts);
int predictive_match_nch_batch(const float* const* refs, const int* strides, int nch, int ntemp, int width, int height,
                               int t0, const SearchConfig& cfg, const MatchBatchItem* items, int count, Match* matches,
                               int match_stride, int* counts);

struct SvdBatchItem {
    int m = 0;
    int n = 0;
    const float* A = nullptr;
    int lda = 0;
    float* U = nullptr;
    int ldu = 0;
    float* S = nullptr;
    float* Vt = nullptr;
    int ldvt = 0;
    float* work = nullptr;
    int work_floats = 0;
    // Optional per-item result: 1 when the item completed, 0 on validation or
    // numerical failure. The function return value remains the first failure.
    int* status = nullptr;
};

// Processes independent SVDs in (m,n) buckets. Return zero on success, or
// first_failed_index + 1 (so item zero is distinguishable from success).
int svd_economy_batch(SvdBatchItem* items, int count);

struct GemmNNBatchItem {
    int m = 0;
    int n = 0;
    int k = 0;
    const float* A = nullptr;
    int lda = 0;
    const float* B = nullptr;
    int ldb = 0;
    float* C = nullptr;
    int ldc = 0;
};

void gemm_nn_batch(GemmNNBatchItem* items, int count);

struct Bm3dFilterBatchItem {
    float* patches = nullptr;
    int lda = 0;
    int group = 0;
    int k = 0;
    int block = 0;
    float sigma = 0.f;
    bool wiener = false;
    const float* ref_patches = nullptr;
    float* weight = nullptr;
    float* work = nullptr;
    int* status = nullptr;
};

// Applies independent BM3D group transforms in stable dimension buckets.
// Returns zero on success, or first_failed_index + 1.
int bm3d_filter_group_batch(Bm3dFilterBatchItem* items, int count);

struct WnnmShrinkBatchItem {
    float* group = nullptr;
    int m = 0;
    int n = 0;
    int lda = 0;
    float sigma = 0.f;
    int residual = 0;
    int adaptive = 0;
    float* adaptive_weight = nullptr;
    float* work = nullptr;
    int work_floats = 0;
    int* status = nullptr;
};

// Applies independent WNNM groups in (m,n) buckets.
int wnnm_shrink_batch(WnnmShrinkBatchItem* items, int count);

struct McwnnmFilterBatchItem {
    float* group = nullptr;
    int m = 0;
    int n = 0;
    int lda = 0;
    int nch = 0;
    const float* sigma = nullptr;
    int admm_iter = 0;
    float rho = 0.f;
    float mu = 0.f;
    int residual = 0;
    int adaptive = 0;
    float* adaptive_weight = nullptr;
    float* work = nullptr;
    int work_floats = 0;
    int* status = nullptr;
};

int mcwnnm_filter_group_batch(McwnnmFilterBatchItem* items, int count);

struct TwscPcaBatchItem {
    float* group = nullptr;
    int m = 0;
    int n = 0;
    int lda = 0;
    float sigma = 0.f;
    float* work = nullptr;
    int work_floats = 0;
    const float* col_sigma = nullptr;
    float* col_weight = nullptr;
    const float* row_weight = nullptr;
    int* status = nullptr;
};

int twsc_pca_soft_batch(TwscPcaBatchItem* items, int count);

struct NcsrFilterBatchItem {
    float* group = nullptr;
    int m = 0;
    int n = 0;
    int lda = 0;
    float sigma = 0.f;
    const float* col_dist = nullptr;
    float* work = nullptr;
    int work_floats = 0;
    int* status = nullptr;
};

int ncsr_filter_group_batch(NcsrFilterBatchItem* items, int count);

struct NlhFilterBatchItem {
    float* patches = nullptr;
    int m = 0;
    int n = 0;
    int lda = 0;
    int q = 0;
    float sigma = 0.f;
    bool wiener = false;
    const float* ref_patches = nullptr;
    float* weight = nullptr;
    float* work = nullptr;
    int work_floats = 0;
    int* status = nullptr;
};

int nlh_filter_group_batch(NlhFilterBatchItem* items, int count);

}  // namespace nss
