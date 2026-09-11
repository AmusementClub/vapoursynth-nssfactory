#pragma once

#include "explore.hpp"
#include "nss/resources.hpp"
#include <chrono>

namespace nss::explore {
template<class T> using Vec = ResourceVector<T>;
using Clock = std::chrono::steady_clock;
extern thread_local std::uint64_t backend_counts[4];
inline double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}
void require(bool condition, const char* message);
void validate_options(const Options& options);
void validate_image(const double* image, int width, int height, int block);

// Owned, per-request dictionary snapshots. No global or cross-frame cache.
// rebuild() invalidates the cast, norms and Gram together after every update.
struct Prepared {
    int m, k, matrix_mode = 0, packed_rows = 0;
    Vec<double> dictionary, gram, norms;
    Vec<float> single, transpose;
    std::uint64_t version = 0;
    Prepared(const double* input, int rows, int atoms);
    void rebuild(int correlation_policy);
};

struct PursuitWork {
    Vec<double> residual, output, correlations, q, dq, r, projections;
    Vec<double> projected_norms, vector, scores, coefficients, history;
    Vec<float> residual32, correlations32, packing32, padding32;
    Vec<int> support;
    Vec<unsigned char> excluded;
    void resize(int m, int k, int g);
};

void correlations(const Prepared& prepared, const double* signals, int count,
                  int precision, double* output, Vec<float>& input32, Vec<float>& result32,
                  Vec<float>& packing32, Vec<float>& padding32);
// Shared FP32 D^T Y adapter for pursuit scores and proximal-learning gradients.
// Packing/padding workspaces must not alias signals or output.
void correlate_float(const Prepared& prepared, const float* signals, int stride, int count,
                     float* output, Vec<float>& packing, Vec<float>& padding);
void solve(const Prepared& prepared, const double* signals, int count, double epsilon,
           const Options& options, PursuitWork& work, Stats& stats,
           const double* initial_correlations = nullptr);

struct GroupPlan { Vec<std::uint64_t> offsets; Vec<int> members; };
GroupPlan group_plan(const double* pilot, int width, int height, int block,
                     const Options& options, Stats& stats);
void pack(const double* image, int width, int block, int nx, const int* members, int count,
          Vec<double>& signals, Vec<double>& means);
void image_run(const double* image, int width, int height, const double* dictionary,
               int block, int atoms, const double* epsilons, int epsilon_count,
               const Options& options, double* output, double* pilot,
               double* learned_dictionary, std::uint32_t* coverage, Stats& stats);

// Explicit penalized alternating-learning approximation, NOT constrained Eq.(8)
// or the author ODL schedule. Singleton L1, then grouped L1,2, sampled from noisy
// data only. Reuses Highway GEMM and the existing soft/prox primitives.
void learn(Prepared& prepared, const double* image, int width, int height, int block,
           const GroupPlan* plan, const Options& options, Stats& stats);
void code_signals(const Prepared& prepared, const double* signals, int count, double lambda,
                  bool grouped, const Options& options, double* coefficients, Stats& stats);
}  // namespace nss::explore
