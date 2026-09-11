#pragma once

#include <cstdint>

// Experimental C ABI, not a VapourSynth filter or a stable product interface.
// Real-valued arrays are packed column-major, except raster image planes.
// No clean/reference image is accepted by any denoising or learning entry.
namespace nss::explore {

struct Options {
    std::int32_t precision = 0;         // 0: FP64; 1: FP32 correlations; 2: FP32 + near-tie refinement
    std::int32_t correlation = 0;       // bits: 1 Gram recurrence, 2 transposed FP32 NN, 4 optional SME NN
    std::int32_t solver = 1;            // 0: Tropp sum-absolute score; 1: OLS energy gain
    std::int32_t grouping = 1;          // 0: disjoint; 1: overlapping members, skip covered seeds
    std::int32_t match_precision = 0;   // 0: FP64 SSD; 1: existing FP32 Highway SSD
    std::int32_t window = 32;
    std::int32_t max_group = 0;         // 0: no algorithmic cap
    std::int32_t max_support = 0;       // 0: rank limit only; nonzero is an explicit approximation
    std::int32_t batch = 128;           // packing/initial-correlation panel, not independent group splitting
    std::int32_t image_passes = 0;
    std::int32_t group_passes = 0;
    std::int32_t learning_samples = 2048;
    std::int32_t learning_groups = 64;
    std::int32_t learning_iterations = 48;
    std::int32_t learning_group_cap = 64;
    std::int32_t reserved = 0;
    double sigma = 25.0 / 255.0;
    double threshold_multiplier = 1.0;
    double epsilon_scale = 1.0;
    double learning_lambda = 0.75;       // lambda = scale*sigma (times sqrt(g) for grouped L1,2)
    double learning_tolerance = 1e-4;
    std::uint64_t memory_limit_bytes = 1024ull * 1024 * 1024;
};

struct Stats {
    std::uint64_t groups = 0, occurrences = 0, pilot_patches = 0;
    std::uint64_t support_sum = 0, support_max = 0, capped_groups = 0;
    std::uint64_t rank_rejections = 0, refined_candidates = 0, correlation_refreshes = 0;
    std::uint64_t dictionary_versions = 0, tracked_peak_bytes = 0;
    std::uint64_t learning_codes = 0, learning_converged = 0, learning_steps = 0;
    std::uint64_t learning_backtracks = 0, dictionary_updates = 0;
    std::uint64_t last_rank = 0, last_history_size = 0, group_max = 0;
    double last_residual = 0, worst_budget_ratio = 0;
    double dictionary_change = 0, learning_stationarity_max = 0;
    double learning_fixed_code_drop = 0;
    double prepare_seconds = 0, image_learning_seconds = 0, pilot_seconds = 0;
    double matching_seconds = 0, group_learning_seconds = 0, final_seconds = 0, total_seconds = 0;
};

}  // namespace nss::explore

#if defined(_WIN32)
#define NSS_EXPLORE_API __declspec(dllexport)
#else
#define NSS_EXPLORE_API __attribute__((visibility("default")))
#endif

extern "C" {
NSS_EXPLORE_API std::uint32_t nss_explore_abi();
NSS_EXPLORE_API std::uint64_t nss_explore_options_size();
NSS_EXPLORE_API std::uint64_t nss_explore_stats_size();
NSS_EXPLORE_API const char* nss_explore_error();
NSS_EXPLORE_API int nss_explore_sme_available();
// Per-thread counters reset at request entry: TN64, TN32, NN32, SME32
// shared D^T Y products (pursuit scores and learning gradients, not Gram preparation).
NSS_EXPLORE_API int nss_explore_backend_counts(std::uint64_t* counts, int count);

// output: m*g; coefficients: k*g (dense diagnostic view); support: min(m,k);
// history: min(m,k)+1. All arrays are caller-owned and must not alias inputs.
NSS_EXPLORE_API int nss_explore_solve(const double* dictionary, const double* signals,
    int m, int k, int g, double epsilon, const nss::explore::Options* options,
    double* output, double* coefficients, int* support, double* history, nss::explore::Stats* stats);

// Diagnostic penalized coding entry: coefficients has k*g FP64 entries holding
// the returned FP32 solution. lambda is the actual penalty, not a noise budget.
NSS_EXPLORE_API int nss_explore_code(const double* dictionary, const double* signals,
    int m, int k, int g, double lambda, int grouped, const nss::explore::Options* options,
    double* coefficients, nss::explore::Stats* stats);

// epsilons[g] is supplied by the existing SciPy chi-square budget implementation.
// Its length must exceed the largest possible group (window^2, or max_group).
// Returned images remain FP64 and are not clipped. coverage is uint32 raster.
NSS_EXPLORE_API int nss_explore_denoise(const double* image, int width, int height,
    const double* dictionary, int block, int atoms, const double* epsilons, int epsilon_count,
    const nss::explore::Options* options, double* output, double* pilot,
    double* learned_dictionary, std::uint32_t* coverage, nss::explore::Stats* stats);

// Diagnostic group-plan export. A null offsets/members pair queries sizes only.
// offsets needs (number_of_valid_patches+1) entries; members needs member_capacity.
NSS_EXPLORE_API int nss_explore_groups(const double* pilot, int width, int height, int block,
    const nss::explore::Options* options, std::uint64_t* offsets, int* members,
    std::uint64_t member_capacity, nss::explore::Stats* stats);
}
