#pragma once

#include "nss/resources.hpp"

namespace nss {

inline constexpr int kTwscMaxRows = 768;
inline constexpr int kTwscMaxColumns = 256;
inline constexpr double kTwscNoiseFloor = 1e-6;

struct TwscSolverOptions {
    int iterations = 10;
    double rho = 0.5;
    double mu = 1.1;
    double tolerance = 1e-6;
};

struct TwscSolverStats {
    int iterations = 0;
    bool converged = false;
    bool double_svd = false;
    double primal = 0;
    double coefficient_change = 0;
    double auxiliary_change = 0;
    double sylvester_residual = 0;
};

// All matrices are column-major. Workspaces own budgeted, reusable storage.
struct TwscWorkspace {
    ResourceVector<float> small, input, qr32, aux32;
    ResourceVector<double> qr64, aux64;
    ResourceVector<double> centered, mean, dictionary, singular, vt, scratch;
    ResourceVector<double> atoms, gram, eigenvectors, eigenvalues, data, coefficients;
    ResourceVector<double> auxiliary, dual, previous, rhs, projected, solution;
};

// Economy SVD, isolated from the existing 256x32 solver contract. A is unchanged.
// The output dictionary has m*min(m,n) entries and vt has min(m,n)*n.
// Uses the existing small solver when possible; checks its residual and retries
// in FP64 before reporting failure. Returns false for a numerical failure.
bool twsc_svd(const float* a, int m, int n, int lda, TwscWorkspace& work, bool& double_fallback,
              bool orthogonal_double = false);
bool twsc_valid_svd(const float* a, int m, int n, int lda, const TwscWorkspace& work);
// Same scalar reduction order, vectorized over independent outputs/dot products.
bool twsc_valid_svd_lanes(const float* a, int m, int n, int lda, const TwscWorkspace& work);
bool twsc_svd_validation_lanes_available();
bool twsc_svd_clustered(const TwscWorkspace& work);
// Prepared finite input groups; independent FP64 lanes with scalar operation order.
void twsc_svd64_batch(TwscWorkspace* const* work, int m, int n, int count);
void twsc_prepare_group(const float* group, int m, int n, int lda, TwscWorkspace& work);

// Symmetric FP64 eigensolver: eigenvectors are columns; input is unchanged.
bool twsc_symmetric_eigen(const double* matrix, int n, double* values, double* vectors,
                          ResourceVector<double>& scratch);
void twsc_gemm_nn(int m, int n, int k, const double* a, const double* b, double* c);
void twsc_gemm_tn(int m, int n, int k, const double* a, const double* b, double* c);

// Independent entry for a fixed dictionary/spectrum. Y and D are unweighted;
// row_precision[i] = 1/max(sigma_channel,1e-6), i.e. W1 squared.
// spectrum is W3 after the selected author noise correction. Output is D*W3*C.
// Optional trace appends C, Z, dual after each iteration (three r*n matrices).
TwscSolverStats twsc_solve(const double* y, const double* dictionary, const double* spectrum,
                          int m, int n, int rank, const double* row_precision,
                          const float* column_sigma, const TwscSolverOptions& options,
                          double* output, TwscWorkspace& work, ResourceVector<double>* trace = nullptr);

// Demean -> unweighted SVD -> noise-correct W3 -> full three-weight ADMM ->
// reconstruct and restore mean. row_sigma has m entries, column_sigma has n.
// Zero channel selection is a host responsibility; zero column noise is floored.
TwscSolverStats twsc_filter_full(float* group, int m, int n, int lda, const float* row_sigma,
                                const float* column_sigma, float* column_weight,
                                const TwscSolverOptions& options, TwscWorkspace& work);

TwscSolverStats twsc_finish_full(float* group, int m, int n, int lda, const float* row_sigma,
                                const float* column_sigma, float* column_weight,
                                const TwscSolverOptions& options, TwscWorkspace& work, bool double_svd);
struct TwscFullBatchItem {
    float* group = nullptr;
    int m = 0, n = 0, lda = 0;
    const float* row_sigma = nullptr;
    const float* column_sigma = nullptr;
    float* column_weight = nullptr;
    const TwscSolverOptions* options = nullptr;
    TwscWorkspace* work = nullptr;
    TwscSolverStats* stats = nullptr;
};
void twsc_filter_full_batch(TwscFullBatchItem* items, int count);

} // namespace nss
