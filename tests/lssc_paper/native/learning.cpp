#include "internal.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_lssc.hpp"
#include "nss/cpu_twsc_full.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nss::explore {
namespace {
std::size_t size(int a, int b) { return checked_mul(std::size_t(a), std::size_t(b)); }
struct CodingWork {
    Vec<float> y, a, extrapolated, next, gradient, reconstruction, next_reconstruction, transpose;
    Vec<float> gram_product, cross_product, preparation;
    Vec<float> correlation_packing, correlation_padding;
    Vec<double> gram_sum, cross_sum;
};

double penalty(const Vec<float>& a, int k, int g, bool grouped) {
    double result = 0;
    if (grouped) {
        for (int row = 0; row < k; ++row) {
            double norm = 0;
            for (int j = 0; j < g; ++j) {
                const double value = a[row + std::size_t(j) * k];
                norm += value * value;
            }
            result += std::sqrt(norm);
        }
    } else for (float value : a) result += std::abs(double(value));
    return result;
}
double fidelity(const Prepared& p, const Vec<float>& a, const Vec<float>& y, int g, Vec<float>& residual) {
    residual.resize(size(p.m, g));
    gemm_nn_hwy(p.m, g, p.k, p.single.data(), p.m, a.data(), p.k, residual.data(), p.m);
    double energy = 0;
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] -= y[i];
        energy += double(residual[i]) * residual[i];
    }
    require(std::isfinite(energy), "nonfinite learning fidelity");
    return 0.5 * energy;
}
void prox(Vec<float>& values, int atoms, int count, double threshold, bool grouped) {
    require(std::isfinite(threshold) && threshold >= 0 && threshold <= std::numeric_limits<float>::max(),
            "invalid proximal threshold");
    if (grouped) lssc_group_soft(values.data(), atoms, count, atoms, static_cast<float>(threshold));
    else soft_threshold(values.data(), checked_int(values.size()), static_cast<float>(threshold));
}

// Monotone-restarted FISTA on an explicitly PENALIZED surrogate. Backtracking
// validates the quadratic majorizer rather than trusting 8 power iterations as
// an upper bound. FP32 kernels, FP64 objective/step checks, reported final PG map.
void code(const Prepared& p, const Vec<double>& signals, int g, double lambda, bool grouped,
          double initial_lipschitz, const Options& o, CodingWork& w, Stats& stats) {
    const int m = p.m, k = p.k;
    const auto coefficients = size(k, g);
    w.y.resize(signals.size());
    for (std::size_t i = 0; i < signals.size(); ++i) {
        w.y[i] = static_cast<float>(signals[i]);
        require(std::isfinite(w.y[i]), "learning signals exceed FP32 range");
    }
    w.a.assign(coefficients, 0); w.extrapolated = w.a;
    w.next.resize(coefficients); w.gradient.resize(coefficients);
    double objective = 0;
    for (float value : w.y) objective += 0.5 * double(value) * value;
    const double initial_objective = objective;
    double lipschitz = std::max(1e-6, initial_lipschitz), momentum = 1;
    for (int iteration = 0; iteration < o.learning_iterations; ++iteration) {
        bool accepted = false;
        for (int restart = 0; restart < 2 && !accepted; ++restart) {
            const double f_y = fidelity(p, w.extrapolated, w.y, g, w.reconstruction);
            correlate_float(p, w.reconstruction.data(), m, g, w.gradient.data(), w.correlation_packing, w.correlation_padding);
            double trial_objective = 0;
            bool majorized = false;
            for (int backtrack = 0; backtrack < 32; ++backtrack) {
                for (std::size_t i = 0; i < coefficients; ++i)
                    w.next[i] = static_cast<float>(double(w.extrapolated[i]) - w.gradient[i] / lipschitz);
                prox(w.next, k, g, lambda / lipschitz, grouped);
                const double f_next = fidelity(p, w.next, w.y, g, w.next_reconstruction);
                double linear = 0, norm_squared = 0;
                for (std::size_t i = 0; i < coefficients; ++i) {
                    const double step = double(w.next[i]) - w.extrapolated[i];
                    linear += step * w.gradient[i]; norm_squared += step * step;
                }
                // Explicit FP32 evaluation allowance, not a convergence claim.
                const double allowance = 2e-6 * std::max(1.0, f_y);
                if (f_next <= f_y + linear + 0.5 * lipschitz * norm_squared + allowance) {
                    majorized = true;
                    trial_objective = f_next + lambda * penalty(w.next, k, g, grouped);
                    break;
                }
                lipschitz *= 2; ++stats.learning_backtracks;
            }
            require(majorized, "learning majorization failed after bounded backtracking");
            if (trial_objective > objective + 2e-6 * std::max(1.0, initial_objective) && restart == 0) {
                w.extrapolated = w.a; momentum = 1;
                continue;
            }
            require(trial_objective <= objective + 4e-6 * std::max(1.0, initial_objective),
                    "learning objective increased after momentum restart");
            const bool proximal_step = momentum == 1;
            const double next_momentum = (1 + std::sqrt(1 + 4 * momentum * momentum)) / 2;
            double change = 0, norm = 0;
            for (std::size_t i = 0; i < coefficients; ++i) {
                const double difference = double(w.next[i]) - w.a[i];
                change += difference * difference; norm += double(w.next[i]) * w.next[i];
                w.extrapolated[i] = static_cast<float>(w.next[i] + (momentum - 1) / next_momentum * difference);
            }
            w.a.swap(w.next); momentum = next_momentum; objective = trial_objective;
            ++stats.learning_steps;
            accepted = true;
            if (lipschitz * std::sqrt(change) / std::max(1.0, std::sqrt(norm)) <= o.learning_tolerance) {
                if (proximal_step) iteration = o.learning_iterations;
                else { w.extrapolated = w.a; momentum = 1; }
                // A small accelerated step alone is not stationarity evidence.
            }
        }
        require(accepted, "learning update was not accepted");
    }
    fidelity(p, w.a, w.y, g, w.reconstruction);
    correlate_float(p, w.reconstruction.data(), m, g, w.gradient.data(), w.correlation_packing, w.correlation_padding);
    for (std::size_t i = 0; i < coefficients; ++i)
        w.next[i] = static_cast<float>(double(w.a[i]) - w.gradient[i] / lipschitz);
    prox(w.next, k, g, lambda / lipschitz, grouped);
    double mapping = 0, norm = 0;
    for (std::size_t i = 0; i < coefficients; ++i) {
        const double step = double(w.next[i]) - w.a[i];
        mapping += step * step; norm += double(w.a[i]) * w.a[i];
    }
    const double stationarity = lipschitz * std::sqrt(mapping) / std::max(1.0, std::sqrt(norm));
    require(std::isfinite(stationarity), "nonfinite learning stationarity");
    ++stats.learning_codes;
    stats.learning_converged += stationarity <= o.learning_tolerance;
    stats.learning_stationarity_max = std::max(stats.learning_stationarity_max, stationarity);
}

void accumulate(const Prepared& p, int g, double weight, CodingWork& w) {
    const int m = p.m, k = p.k;
    w.transpose.resize(size(g, k));
    for (int atom = 0; atom < k; ++atom) for (int j = 0; j < g; ++j)
        w.transpose[j + std::size_t(atom) * g] = w.a[atom + std::size_t(j) * k];
    w.gram_product.resize(size(k, k)); w.cross_product.resize(size(m, k));
    gemm_nn_hwy(k, k, g, w.a.data(), k, w.transpose.data(), g, w.gram_product.data(), k);
    gemm_nn_hwy(m, k, g, w.y.data(), m, w.transpose.data(), g, w.cross_product.data(), m);
    for (std::size_t i = 0; i < w.gram_product.size(); ++i) w.gram_sum[i] += weight * w.gram_product[i];
    for (std::size_t i = 0; i < w.cross_product.size(); ++i) w.cross_sum[i] += weight * w.cross_product[i];
}

double dictionary_objective(const Vec<double>& dictionary, const CodingWork& w, int m, int k,
                            Vec<double>& product) {
    product.resize(size(m, k));
    twsc_gemm_nn(m, k, k, dictionary.data(), w.gram_sum.data(), product.data());
    double value = 0;
    for (std::size_t i = 0; i < dictionary.size(); ++i)
        value += 0.5 * dictionary[i] * product[i] - dictionary[i] * w.cross_sum[i];
    return value;
}

void update(Prepared& p, CodingWork& w, const Options& o, Stats& stats) {
    const int m = p.m, k = p.k;
    for (int j = 0; j < k; ++j) for (int i = 0; i < j; ++i) {
        const double mean = 0.5 * (w.gram_sum[i + std::size_t(j) * k] + w.gram_sum[j + std::size_t(i) * k]);
        w.gram_sum[i + std::size_t(j) * k] = w.gram_sum[j + std::size_t(i) * k] = mean;
    }
    Vec<double> product, column(m);
    const double before = dictionary_objective(p.dictionary, w, m, k, product);
    for (int atom = 0; atom < k; ++atom) {
        const double diagonal = w.gram_sum[atom + std::size_t(atom) * k];
        if (diagonal <= 1e-14) continue;  // Unused atoms are retained, never silently randomized.
        twsc_gemm_nn(m, 1, k, p.dictionary.data(), w.gram_sum.data() + std::size_t(atom) * k, column.data());
        double norm = 0;
        for (int row = 0; row < m; ++row) {
            const auto at = row + std::size_t(atom) * m;
            column[row] = p.dictionary[at] + (w.cross_sum[at] - column[row]) / diagonal;
            norm += column[row] * column[row];
        }
        const double scale = 1 / std::max(1.0, std::sqrt(norm));
        for (int row = 0; row < m; ++row) p.dictionary[row + std::size_t(atom) * m] = column[row] * scale;
    }
    const double after = dictionary_objective(p.dictionary, w, m, k, product);
    require(std::isfinite(after) && after <= before + 1e-9 * std::max(1.0, std::abs(before)),
            "dictionary update increased its fixed-code surrogate");
    stats.learning_fixed_code_drop += before - after;
    ++stats.dictionary_updates;
    // Norms need not equal one after projection onto the unit BALL. Pursuit
    // uses actual norms and rank checks, not a hidden unit-norm assumption.
    p.rebuild(o.correlation);
}

int sample(int index, int wanted, int total) {
    return static_cast<int>(((2ull * std::uint64_t(index) + 1) * total) / (2ull * wanted));
}
}  // namespace

void code_signals(const Prepared& p, const double* signals, int count, double lambda,
                  bool grouped, const Options& o, double* coefficients, Stats& stats) {
    require(signals && coefficients && count >= 1 && count <= 16384 &&
            std::isfinite(lambda) && lambda >= 0, "invalid penalized coding request");
    Vec<double> values(signals, signals + size(p.m, count));
    CodingWork work;
    work.preparation.resize(lssc_prepare_work_floats(p.m, p.k));
    LsscPreparedContext context;
    require(lssc_prepare_context(p.single.data(), p.m, p.k, p.m, work.preparation.data(),
            checked_int(work.preparation.size()), &context) == 0, "coding Lipschitz preparation failed");
    code(p, values, count, lambda, grouped, context.lipschitz, o, work, stats);
    for (std::size_t i = 0; i < work.a.size(); ++i) coefficients[i] = work.a[i];
}

void learn(Prepared& p, const double* image, int width, int height, int block,
           const GroupPlan* plan, const Options& o, Stats& stats) {
    const bool grouped = plan != nullptr;
    const int passes = grouped ? o.group_passes : o.image_passes;
    const int nx = width - block + 1, total = nx * (height - block + 1);
    const int wanted = grouped ? std::min(o.learning_groups, checked_int(plan->offsets.size() - 1))
                               : std::min(o.learning_samples, total);
    CodingWork work;
    Vec<int> members;
    Vec<double> signals, means;
    for (int pass = 0; pass < passes; ++pass) {
        work.gram_sum.assign(size(p.k, p.k), 0.0); work.cross_sum.assign(size(p.m, p.k), 0.0);
        work.preparation.resize(lssc_prepare_work_floats(p.m, p.k));
        LsscPreparedContext context;
        require(lssc_prepare_context(p.single.data(), p.m, p.k, p.m, work.preparation.data(),
                checked_int(work.preparation.size()), &context) == 0, "learning Lipschitz preparation failed");
        for (int first = 0; first < wanted;) {
            int count;
            if (grouped) {
                const int group = sample(first, wanted, checked_int(plan->offsets.size() - 1));
                const auto begin = plan->offsets[group];
                const int available = checked_int(plan->offsets[group + 1] - begin);
                count = std::min(available, o.learning_group_cap);
                members.resize(count);
                for (int j = 0; j < count; ++j) members[j] = plan->members[begin + sample(j, count, available)];
                ++first;
            } else {
                count = std::min(o.batch, wanted - first);
                members.resize(count);
                for (int j = 0; j < count; ++j) members[j] = sample(first + j, wanted, total);
                first += count;
            }
            pack(image, width, block, nx, members.data(), count, signals, means);
            const double lambda = o.learning_lambda * o.sigma * (grouped ? std::sqrt(double(count)) : 1.0);
            code(p, signals, count, lambda, grouped, context.lipschitz, o, work, stats);
            // Image sample observations have equal weight. Grouped surrogate
            // averages each sampled group's data term, giving groups equal weight.
            accumulate(p, count, grouped ? 1.0 / count : 1.0, work);
        }
        update(p, work, o, stats);
    }
}
}  // namespace nss::explore
