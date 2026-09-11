#include "internal.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"
#include "nss/cpu_twsc_full.hpp"
#include "cpu/bm/matcher.hpp"
#include "cpu/lssc/gemm.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace nss::explore {
thread_local std::uint64_t backend_counts[4] = {};
namespace {
constexpr double unit = std::numeric_limits<double>::epsilon();
std::size_t size(int a, int b) { return checked_mul(std::size_t(a), std::size_t(b)); }
double squared(const double* values, std::size_t count) {
    double sum = 0;
    for (std::size_t i = 0; i < count; ++i) sum += values[i] * values[i];
    return sum;
}
double dot(const double* a, const double* b, int count) {
    double sum = 0;
    for (int i = 0; i < count; ++i) sum += a[i] * b[i];
    return sum;
}
double score(const double* values, int atom, int atoms, int count, int solver, double norm) {
    double sum = 0;
    for (int j = 0; j < count; ++j) {
        const double value = values[atom + std::size_t(j) * atoms];
        sum += solver ? value * value : std::abs(value);
    }
    return solver ? sum / norm : sum;
}
void aggregate(const double* patches, const double* means, int count, int block,
               const int* members, int nx, int width, Vec<double>& numerator,
               Vec<std::uint32_t>& coverage) {
    const int m = block * block;
    for (int j = 0; j < count; ++j) {
        const int y = members[j] / nx, x = members[j] % nx;
        for (int dx = 0; dx < block; ++dx) for (int dy = 0; dy < block; ++dy) {
            const std::size_t pixel = std::size_t(y + dy) * width + x + dx;
            numerator[pixel] += patches[dy + dx * block + std::size_t(j) * m] + means[j];
            require(coverage[pixel] != std::numeric_limits<std::uint32_t>::max(), "coverage overflow");
            ++coverage[pixel];
        }
    }
}
void finish(const Vec<double>& numerator, const Vec<std::uint32_t>& coverage, double* output) {
    for (std::size_t i = 0; i < numerator.size(); ++i) {
        require(coverage[i] != 0, "uncovered pixel (no source fallback)");
        output[i] = numerator[i] / coverage[i];
        require(std::isfinite(output[i]), "nonfinite aggregated output");
    }
}
}  // namespace

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void validate_options(const Options& o) {
    require(o.precision >= 0 && o.precision <= 2 && o.correlation >= 0 && o.correlation <= 7 &&
            (!(o.correlation & 4) || (o.correlation & 2)),
            "invalid precision/correlation policy");
    require(o.solver >= 0 && o.solver <= 1 && o.grouping >= 0 && o.grouping <= 1 &&
            o.match_precision >= 0 && o.match_precision <= 1, "invalid solver/grouping/matching policy");
    require(o.window >= 1 && o.window <= 128 && o.batch >= 1 && o.batch <= 512,
            "invalid window or batch");
    require(o.max_group >= 0 && o.max_group <= o.window * o.window &&
            o.max_support >= 0 && o.max_support <= 1024, "invalid group/support cap");
    require(o.image_passes >= 0 && o.image_passes <= 100 && o.group_passes >= 0 && o.group_passes <= 100,
            "invalid learning passes");
    require(o.learning_samples >= 1 && o.learning_groups >= 1 && o.learning_iterations >= 1 &&
            o.learning_iterations <= 10000 && o.learning_group_cap >= 1, "invalid learning budget");
    require(std::isfinite(o.sigma) && o.sigma >= 0 && std::isfinite(o.threshold_multiplier) &&
            o.threshold_multiplier > 0 && std::isfinite(o.epsilon_scale) && o.epsilon_scale > 0,
            "invalid noise/threshold/residual budget");
    require(std::isfinite(o.learning_lambda) && o.learning_lambda >= 0 &&
            std::isfinite(o.learning_tolerance) && o.learning_tolerance > 0 &&
            o.memory_limit_bytes >= 1024 && o.reserved == 0, "invalid learning tolerance/memory/ABI");
}
void validate_image(const double* image, int width, int height, int block) {
    require(image && block >= 1 && block <= 32 && width >= block && height >= block,
            "invalid image geometry");
    const auto count = size(width, height);
    require(count <= std::size_t(std::numeric_limits<int>::max()), "image exceeds index contract");
    for (std::size_t i = 0; i < count; ++i) require(std::isfinite(image[i]), "nonfinite image");
}

Prepared::Prepared(const double* input, int rows, int atoms) : m(rows), k(atoms) {
    require(input && m >= 1 && m <= 1024 && k >= 1 && k <= 4096, "invalid dictionary dimensions");
    dictionary.assign(input, input + size(m, k));
    for (int j = 0; j < k; ++j) {
        const double norm = squared(dictionary.data() + std::size_t(j) * m, m);
        require(std::isfinite(norm) && std::abs(std::sqrt(norm) - 1) <= 1.01e-8,
                "dictionary columns must be finite and unit norm; no implicit normalization");
    }
}
void Prepared::rebuild(int correlation_policy) {
    matrix_mode = correlation_policy & 6;
    packed_rows = (matrix_mode & 4) && m >= 64 && m <= 256 && lssc_sme_available()
                      ? (m + 15) / 16 * 16 : m;
    single.resize(dictionary.size());
    norms.resize(k);
    for (std::size_t i = 0; i < dictionary.size(); ++i) {
        require(std::isfinite(dictionary[i]), "nonfinite updated dictionary");
        single[i] = static_cast<float>(dictionary[i]);
        require(std::isfinite(single[i]), "dictionary exceeds FP32 range");
    }
    for (int j = 0; j < k; ++j) norms[j] = squared(dictionary.data() + std::size_t(j) * m, m);
    if (matrix_mode) {
        transpose.assign(size(k, packed_rows), 0.f);
        lssc_pack_columns_hwy(single.data(), m, k, m, transpose.data());
    } else transpose.clear();
    if (correlation_policy & 1) {
        gram.resize(size(k, k));
        twsc_gemm_tn(m, k, k, dictionary.data(), dictionary.data(), gram.data());
    } else gram.clear();
    ++version;
}

void correlate_float(const Prepared& p, const float* signals, int stride, int count,
                     float* output, Vec<float>& packing, Vec<float>& padding) {
    const bool use_sme = (p.matrix_mode & 4) && count >= 128 && p.m >= 64 && p.m <= 256 &&
                        (p.k >= 256 || (p.k >= 64 && p.k % 16 == 0)) && lssc_sme_available();
    if (use_sme) {
        const int rows = p.packed_rows;
        const float* input = signals;
        int leading = stride;
        if (rows != p.m) {
            padding.resize(size(rows, count));
            for (int j = 0; j < count; ++j) {
                std::copy_n(signals + std::size_t(j) * stride, p.m, padding.data() + std::size_t(j) * rows);
                std::fill_n(padding.data() + std::size_t(j) * rows + p.m, rows - p.m, 0.f);
            }
            input = padding.data(); leading = rows;
        }
        // Split output atoms, not the mathematical dictionary/support/group.
        // Every row still sees every logical patch pixel. Padding is confined
        // to the multiply, never to centering, chi-square degrees or QR.
        for (int first = 0; first < p.k; first += 256) {
            const int atoms = std::min(p.k - first, 256);
            packing.resize(lssc_gemm_pack_work_floats(atoms, count, rows));
            const bool used = lssc_gemm_nn(atoms, count, rows, p.transpose.data() + first, p.k,
                input, leading, output + first, p.k, packing.data(), checked_int(packing.size()));
            if (used) backend_counts[3] += (count + 255) / 256;
            else ++backend_counts[2];
        }
    } else if (p.matrix_mode) {
        // Short products and unsupported hardware keep the logical row count;
        // do not pay SME padding/splitting cost on the ordinary Highway path.
        gemm_nn_hwy(p.k, count, p.m, p.transpose.data(), p.k, signals, stride, output, p.k);
        ++backend_counts[2];
    } else {
        gemm_tn_hwy(p.m, count, p.k, p.single.data(), p.m, signals, stride, output, p.k);
        ++backend_counts[1];
    }
}

void correlations(const Prepared& p, const double* signals, int count, int precision,
                  double* output, Vec<float>& input32, Vec<float>& result32,
                  Vec<float>& packing32, Vec<float>& padding32) {
    if (!precision) {
        twsc_gemm_tn(p.m, count, p.k, p.dictionary.data(), signals, output);
        ++backend_counts[0];
        return;
    }
    input32.resize(size(p.m, count));
    result32.resize(size(p.k, count));
    for (std::size_t i = 0; i < input32.size(); ++i) {
        input32[i] = static_cast<float>(signals[i]);
        require(std::isfinite(input32[i]), "signals exceed FP32 range");
    }
    correlate_float(p, input32.data(), p.m, count, result32.data(), packing32, padding32);
    for (std::size_t i = 0; i < result32.size(); ++i) {
        require(std::isfinite(result32[i]), "nonfinite FP32 correlation");
        output[i] = result32[i];
    }
}

void PursuitWork::resize(int m, int k, int g) {
    const int limit = std::min(m, k);
    residual.resize(size(m, g)); output.resize(size(m, g)); correlations.resize(size(k, g));
    q.resize(size(m, limit)); dq.resize(size(k, limit)); r.resize(size(limit, limit));
    projections.resize(size(limit, g)); projected_norms.resize(k); vector.resize(m); scores.resize(k);
    excluded.assign(k, 0); support.clear(); history.clear();
    support.reserve(limit); history.reserve(limit + 1);
}

void solve(const Prepared& p, const double* signals, int g, double epsilon,
           const Options& o, PursuitWork& w, Stats& stats, const double* initial_correlations) {
    require(signals && g >= 1 && g <= 16384 && std::isfinite(epsilon) && epsilon >= 0,
            "invalid pursuit signals or budget");
    require(!(o.correlation & 1) || p.gram.size() == size(p.k, p.k), "stale/missing Gram cache");
    require(p.matrix_mode == (o.correlation & 6), "stale dictionary packing policy");
    const bool recurrence = (o.correlation & 1) != 0;
    const int m = p.m, k = p.k, limit = std::min(m, k);
    const auto count = size(m, g);
    w.resize(m, k, g);
    std::copy_n(signals, count, w.residual.data());
    std::fill(w.output.begin(), w.output.end(), 0.0);
    std::copy(p.norms.begin(), p.norms.end(), w.projected_norms.begin());
    double energy = squared(signals, count);
    require(std::isfinite(energy), "nonfinite signal energy");
    const double initial_energy = energy, tolerance = 64 * unit * std::max(1.0, energy);
    const int cap = o.max_support ? std::min(limit, o.max_support) : limit;
    w.history.push_back(energy);
    if (initial_correlations) std::copy_n(initial_correlations, size(k, g), w.correlations.data());
    else if (energy > epsilon + tolerance)
        correlations(p, signals, g, o.precision, w.correlations.data(), w.residual32, w.correlations32, w.packing32, w.padding32);
    int rejected = 0;
    while (energy > epsilon + tolerance && int(w.support.size()) < cap && rejected < k) {
        const int rank = int(w.support.size());
        if (!recurrence && rank > 0)
            correlations(p, w.residual.data(), g, o.precision, w.correlations.data(), w.residual32, w.correlations32, w.packing32, w.padding32);
        int atom = -1;
        double best = -1;
        for (int a = 0; a < k; ++a) {
            if (w.excluded[a] || (o.solver && w.projected_norms[a] <= 128 * unit)) {
                w.scores[a] = -1;
                continue;
            }
            const double value = score(w.correlations.data(), a, k, g, o.solver, w.projected_norms[a]);
            require(std::isfinite(value), "nonfinite atom score");
            w.scores[a] = value;
            if (value > best) { best = value; atom = a; }
        }
        if (atom < 0) break;
        // This is an explicit near-tie heuristic, not a certified bit-exact
        // mixed-precision selector. Refits and feasibility remain FP64.
        if (o.precision == 2) {
            const double corr_error = (8.0 * m + 32.0 * rank + 64) * 0x1p-24 * std::sqrt(initial_energy);
            auto exact_score = [&](int candidate) {
                double result = 0;
                for (int j = 0; j < g; ++j) {
                    const double value = dot(p.dictionary.data() + std::size_t(candidate) * m,
                                             w.residual.data() + std::size_t(j) * m, m);
                    result += o.solver ? value * value : std::abs(value);
                }
                ++stats.refined_candidates;
                return o.solver ? result / w.projected_norms[candidate] : result;
            };
            best = exact_score(atom);
            for (int a = 0; a < k; ++a) {
                if (a == atom || w.scores[a] < 0) continue;
                const double allowance = o.solver
                    ? (2 * std::sqrt(std::max(0.0, w.scores[a] * w.projected_norms[a]) * g) * corr_error +
                       g * corr_error * corr_error) / w.projected_norms[a]
                    : g * corr_error;
                if (w.scores[a] + allowance < best) continue;
                const double value = exact_score(a);
                if (value > best || (value == best && a < atom)) { best = value; atom = a; }
            }
        }
        if (best <= (o.solver ? tolerance : unit * std::max(1.0, std::sqrt(initial_energy)))) break;
        w.excluded[atom] = 1;
        std::copy_n(p.dictionary.data() + std::size_t(atom) * m, m, w.vector.data());
        for (int i = 0; i < rank; ++i) w.r[i + std::size_t(rank) * limit] = 0;
        // Twice-reorthogonalized modified Gram-Schmidt. The active basis is
        // shared by every RHS, with no fixed 8- or 32-column storage contract.
        for (int pass = 0; pass < 2; ++pass) for (int i = 0; i < rank; ++i) {
            const double* basis = w.q.data() + std::size_t(i) * m;
            const double projection = dot(basis, w.vector.data(), m);
            w.r[i + std::size_t(rank) * limit] += projection;
            for (int row = 0; row < m; ++row) w.vector[row] -= basis[row] * projection;
        }
        const double norm_squared = squared(w.vector.data(), m);
        if (norm_squared <= 128 * unit) {
            ++rejected; ++stats.rank_rejections;
            continue;
        }
        const double norm = std::sqrt(norm_squared);
        w.r[rank + std::size_t(rank) * limit] = norm;
        double* basis = w.q.data() + std::size_t(rank) * m;
        for (int row = 0; row < m; ++row) basis[row] = w.vector[row] / norm;
        double* projected_dictionary = w.dq.data() + std::size_t(rank) * k;
        if (recurrence && norm > 1e-4 && rank % 16 != 15) {
            for (int a = 0; a < k; ++a) {
                double value = p.gram[a + std::size_t(atom) * k];
                for (int i = 0; i < rank; ++i)
                    value -= w.dq[a + std::size_t(i) * k] * w.r[i + std::size_t(rank) * limit];
                projected_dictionary[a] = value / norm;
            }
        } else {
            twsc_gemm_tn(m, 1, k, p.dictionary.data(), basis, projected_dictionary);
            if (recurrence) ++stats.correlation_refreshes;
        }
        for (int a = 0; a < k; ++a)
            w.projected_norms[a] = std::max(0.0, w.projected_norms[a] - projected_dictionary[a] * projected_dictionary[a]);
        for (int j = 0; j < g; ++j) {
            const double projection = dot(basis, signals + std::size_t(j) * m, m);
            w.projections[rank + std::size_t(j) * limit] = projection;
            for (int row = 0; row < m; ++row) {
                const auto index = row + std::size_t(j) * m;
                w.output[index] += basis[row] * projection;
                w.residual[index] = signals[index] - w.output[index];
            }
            if (recurrence) for (int a = 0; a < k; ++a)
                w.correlations[a + std::size_t(j) * k] -= projected_dictionary[a] * projection;
        }
        const double next_energy = squared(w.residual.data(), count);
        require(std::isfinite(next_energy) && next_energy <= energy + tolerance, "QR refit increased residual");
        energy = next_energy;
        w.support.push_back(atom); w.history.push_back(energy);
        // Periodic re-anchoring prevents a long recurrence from deciding a
        // tiny residual using only cancellation-prone original correlations.
        if (recurrence && int(w.support.size()) % 16 == 0 && energy > epsilon + tolerance) {
            correlations(p, w.residual.data(), g, o.precision, w.correlations.data(), w.residual32, w.correlations32, w.packing32, w.padding32);
            ++stats.correlation_refreshes;
        }
    }
    const int rank = int(w.support.size());
    if (energy > epsilon + tolerance) {
        require(o.max_support > 0 && rank == cap && cap < limit, "dictionary span/numerical rank exhausted before noise budget");
        ++stats.capped_groups;
    }
    w.coefficients.resize(size(rank, g));
    for (int j = 0; j < g; ++j) for (int i = rank - 1; i >= 0; --i) {
        double value = w.projections[i + std::size_t(j) * limit];
        for (int t = i + 1; t < rank; ++t)
            value -= w.r[i + std::size_t(t) * limit] * w.coefficients[t + std::size_t(j) * rank];
        w.coefficients[i + std::size_t(j) * rank] = value / w.r[i + std::size_t(i) * limit];
    }
    stats.last_rank = rank; stats.last_history_size = w.history.size(); stats.last_residual = energy;
    stats.support_sum += rank; stats.support_max = std::max(stats.support_max, std::uint64_t(rank));
    stats.worst_budget_ratio = std::max(stats.worst_budget_ratio, energy / std::max(epsilon, tolerance));
}

void pack(const double* image, int width, int block, int nx, const int* members, int count,
          Vec<double>& signals, Vec<double>& means) {
    const int m = block * block;
    signals.resize(size(m, count)); means.resize(count);
    for (int j = 0; j < count; ++j) {
        const int y = members[j] / nx, x = members[j] % nx;
        double mean = 0;
        for (int dx = 0; dx < block; ++dx) for (int dy = 0; dy < block; ++dy) {
            const double value = image[std::size_t(y + dy) * width + x + dx];
            signals[dy + dx * block + std::size_t(j) * m] = value;
            mean += value;
        }
        mean /= m; means[j] = mean;
        for (int row = 0; row < m; ++row) signals[row + std::size_t(j) * m] -= mean;
    }
}

GroupPlan group_plan(const double* pilot, int width, int height, int block, const Options& o, Stats& stats) {
    const int nx = width - block + 1, ny = height - block + 1, patches = nx * ny;
    const double threshold = std::pow(32 * o.sigma, 2) / (block * block) * o.threshold_multiplier;
    require(std::isfinite(threshold), "nonfinite matching threshold");
    GroupPlan plan;
    plan.offsets.push_back(0);
    Vec<unsigned char> covered(patches, 0);
    Vec<float> pilot32;
    if (o.match_precision) {
        pilot32.resize(size(width, height));
        for (std::size_t i = 0; i < pilot32.size(); ++i) {
            pilot32[i] = static_cast<float>(pilot[i]);
            require(std::isfinite(pilot32[i]), "pilot exceeds FP32 range");
        }
    }
    Vec<Match> matches(o.max_group);
    Vec<int> selected;
    selected.reserve(o.max_group);
    for (int seed = 0; seed < patches; ++seed) {
        if (covered[seed]) continue;
        const int sy = seed / nx, sx = seed % nx;
        detail::StableTopK top(matches.data(), o.max_group);
        const auto begin = plan.members.size();
        for (int y = std::max(0, sy - o.window / 2); y < std::min(ny, sy + (o.window - 1) / 2 + 1); ++y)
            for (int x = std::max(0, sx - o.window / 2); x < std::min(nx, sx + (o.window - 1) / 2 + 1); ++x) {
                const int candidate = y * nx + x;
                if (!o.grouping && covered[candidate]) continue;
                double distance = 0;
                if (o.match_precision)
                    distance = ssd_block(pilot32.data() + sy * width + sx, width,
                                         pilot32.data() + y * width + x, width, block);
                else for (int dx = 0; dx < block; ++dx) for (int dy = 0; dy < block; ++dy) {
                    const double difference = pilot[std::size_t(sy + dy) * width + sx + dx] -
                                              pilot[std::size_t(y + dy) * width + x + dx];
                    distance += difference * difference;
                }
                if (distance > threshold) continue;
                if (o.max_group) top.add({x, y, 0, static_cast<float>(distance),
                                          candidate == seed ? 0u : std::uint32_t(candidate + 1)});
                else plan.members.push_back(candidate);
            }
        if (o.max_group) {
            const int count = top.finish();
            selected.resize(count);
            for (int i = 0; i < count; ++i) selected[i] = matches[i].y * nx + matches[i].x;
            std::sort(selected.begin(), selected.end());
            plan.members.insert(plan.members.end(), selected.begin(), selected.end());
        }
        bool self = false;
        for (std::size_t i = begin; i < plan.members.size(); ++i) {
            const int member = plan.members[i];
            covered[member] = 1; self |= member == seed;
        }
        require(self, "group omitted its seed");
        stats.group_max = std::max(stats.group_max, std::uint64_t(plan.members.size() - begin));
        plan.offsets.push_back(plan.members.size());
    }
    stats.groups = plan.offsets.size() - 1; stats.occurrences = plan.members.size();
    return plan;
}

void image_run(const double* image, int width, int height, const double* dictionary,
               int block, int atoms, const double* epsilons, int epsilon_count, const Options& o,
               double* output, double* pilot, double* learned_dictionary, std::uint32_t* coverage, Stats& stats) {
    const auto total_begin = Clock::now();
    validate_image(image, width, height, block);
    require(output && pilot && learned_dictionary && coverage && epsilons, "null image output/budget");
    const int maximum_group = o.max_group ? o.max_group : o.window * o.window;
    require(epsilon_count > maximum_group, "incomplete chi-square budget table");
    for (int g = 1; g <= maximum_group; ++g)
        require(std::isfinite(epsilons[g]) && epsilons[g] >= 0, "invalid chi-square budget table");
    const auto pixels = size(width, height);
    const int nx = width - block + 1, ny = height - block + 1, patches = nx * ny, m = block * block;
    auto begin = Clock::now();
    Prepared prepared(dictionary, m, atoms);
    prepared.rebuild(o.correlation);
    stats.prepare_seconds = elapsed(begin);
    begin = Clock::now();
    if (o.image_passes) learn(prepared, image, width, height, block, nullptr, o, stats);
    stats.image_learning_seconds = elapsed(begin);
    PursuitWork work;
    Vec<double> signals, means, initial, numerator(pixels, 0.0);
    Vec<float> packed32, corr32, packing32, padding32;
    Vec<int> members(o.batch);
    Vec<std::uint32_t> counts(pixels, 0);
    begin = Clock::now();
    for (int first = 0; first < patches;) {
        const int count = std::min(o.batch, patches - first);
        for (int j = 0; j < count; ++j) members[j] = first + j;
        pack(image, width, block, nx, members.data(), count, signals, means);
        initial.resize(size(atoms, count));
        correlations(prepared, signals.data(), count, o.precision, initial.data(), packed32, corr32, packing32, padding32);
        for (int j = 0; j < count; ++j) {
            solve(prepared, signals.data() + std::size_t(j) * m, 1, epsilons[1] * o.epsilon_scale,
                  o, work, stats, initial.data() + std::size_t(j) * atoms);
            aggregate(work.output.data(), means.data() + j, 1, block, members.data() + j, nx, width, numerator, counts);
        }
        first += count;
    }
    finish(numerator, counts, pilot);
    stats.pilot_patches = patches; stats.pilot_seconds = elapsed(begin);
    begin = Clock::now();
    const GroupPlan plan = group_plan(pilot, width, height, block, o, stats);
    stats.matching_seconds = elapsed(begin);
    begin = Clock::now();
    if (o.group_passes) learn(prepared, image, width, height, block, &plan, o, stats);
    stats.group_learning_seconds = elapsed(begin);
    begin = Clock::now();
    std::fill(numerator.begin(), numerator.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);
    for (std::size_t group = 0; group + 1 < plan.offsets.size(); ++group) {
        const auto offset = plan.offsets[group];
        const int count = checked_int(plan.offsets[group + 1] - offset);
        const int* group_members = plan.members.data() + offset;
        pack(image, width, block, nx, group_members, count, signals, means);
        solve(prepared, signals.data(), count, epsilons[count] * o.epsilon_scale, o, work, stats);
        aggregate(work.output.data(), means.data(), count, block, group_members, nx, width, numerator, counts);
    }
    finish(numerator, counts, output);
    std::copy(counts.begin(), counts.end(), coverage);
    std::copy(prepared.dictionary.begin(), prepared.dictionary.end(), learned_dictionary);
    stats.dictionary_versions = prepared.version;
    for (std::size_t i = 0; i < prepared.dictionary.size(); ++i)
        stats.dictionary_change = std::max(stats.dictionary_change, std::abs(prepared.dictionary[i] - dictionary[i]));
    stats.final_seconds = elapsed(begin); stats.total_seconds = elapsed(total_begin);
}
}  // namespace nss::explore
