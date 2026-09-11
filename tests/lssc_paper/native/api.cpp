#include "internal.hpp"
#include "nss/cpu_lssc.hpp"

#include <algorithm>
#include <exception>
#include <string>

namespace {
thread_local std::string last_error;
template<class Function>
int invoke(const nss::explore::Options* options, nss::explore::Stats* stats, Function function) {
    std::shared_ptr<nss::ResourceBudget> budget;
    try {
        last_error.clear();
        std::fill_n(nss::explore::backend_counts, 4, 0);
        nss::explore::require(stats && options, "null options/stats");
        *stats = {};
        const auto& o = *options;
        nss::explore::validate_options(o);
        budget = std::make_shared<nss::ResourceBudget>(o.memory_limit_bytes);
        nss::ResourceScope scope(budget);
        function(o, *stats);
        stats->tracked_peak_bytes = budget->snapshot().peak;
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
    } catch (...) {
        last_error = "unknown native exploration failure";
    }
    if (stats && budget) stats->tracked_peak_bytes = budget->snapshot().peak;
    return -1;
}
}

extern "C" {
std::uint32_t nss_explore_abi() { return 1; }
std::uint64_t nss_explore_options_size() { return sizeof(nss::explore::Options); }
std::uint64_t nss_explore_stats_size() { return sizeof(nss::explore::Stats); }
const char* nss_explore_error() { return last_error.c_str(); }
int nss_explore_sme_available() { return nss::lssc_sme_available(); }
int nss_explore_backend_counts(std::uint64_t* counts, int count) {
    if (!counts || count < 4) return -1;
    std::copy_n(nss::explore::backend_counts, 4, counts);
    return 0;
}

int nss_explore_solve(const double* dictionary, const double* signals, int m, int k, int g, double epsilon,
                     const nss::explore::Options* options, double* output, double* coefficients,
                     int* support, double* history, nss::explore::Stats* stats) {
    return invoke(options, stats, [&](const auto& o, auto& s) {
        nss::explore::require(output && coefficients && support && history, "null pursuit output");
        const auto begin = nss::explore::Clock::now();
        nss::explore::Prepared p(dictionary, m, k);
        p.rebuild(o.correlation);
        nss::explore::PursuitWork work;
        nss::explore::solve(p, signals, g, epsilon, o, work, s);
        std::copy(work.output.begin(), work.output.end(), output);
        std::fill_n(coefficients, nss::checked_mul(std::size_t(k), std::size_t(g)), 0.0);
        const auto rank = work.support.size();
        for (int j = 0; j < g; ++j) for (std::size_t i = 0; i < rank; ++i)
            coefficients[work.support[i] + std::size_t(j) * k] = work.coefficients[i + std::size_t(j) * rank];
        std::copy(work.support.begin(), work.support.end(), support);
        std::copy(work.history.begin(), work.history.end(), history);
        s.dictionary_versions = p.version;
        s.total_seconds = nss::explore::elapsed(begin);
    });
}

int nss_explore_denoise(const double* image, int width, int height, const double* dictionary,
                       int block, int atoms, const double* epsilons, int epsilon_count,
                       const nss::explore::Options* options, double* output, double* pilot,
                       double* learned_dictionary, std::uint32_t* coverage, nss::explore::Stats* stats) {
    return invoke(options, stats, [&](const auto& o, auto& s) {
        nss::explore::image_run(image, width, height, dictionary, block, atoms, epsilons, epsilon_count,
                               o, output, pilot, learned_dictionary, coverage, s);
    });
}

int nss_explore_code(const double* dictionary, const double* signals, int m, int k, int g,
                     double lambda, int grouped, const nss::explore::Options* options,
                     double* coefficients, nss::explore::Stats* stats) {
    return invoke(options, stats, [&](const auto& o, auto& s) {
        nss::explore::require(grouped == 0 || grouped == 1, "invalid coding penalty");
        const auto begin = nss::explore::Clock::now();
        nss::explore::Prepared p(dictionary, m, k);
        p.rebuild(o.correlation & 6);
        nss::explore::code_signals(p, signals, g, lambda, grouped != 0, o, coefficients, s);
        s.dictionary_versions = p.version;
        s.total_seconds = nss::explore::elapsed(begin);
    });
}

int nss_explore_groups(const double* pilot, int width, int height, int block,
                      const nss::explore::Options* options, std::uint64_t* offsets, int* members,
                      std::uint64_t member_capacity, nss::explore::Stats* stats) {
    return invoke(options, stats, [&](const auto& o, auto& s) {
        nss::explore::validate_image(pilot, width, height, block);
        const auto plan = nss::explore::group_plan(pilot, width, height, block, o, s);
        if (!offsets && !members) return;
        nss::explore::require(offsets && members && member_capacity >= plan.members.size(), "group export capacity");
        std::copy(plan.offsets.begin(), plan.offsets.end(), offsets);
        std::copy(plan.members.begin(), plan.members.end(), members);
    });
}
}
