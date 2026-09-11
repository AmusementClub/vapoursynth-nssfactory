#pragma once

// Experimental only: copied into an isolated candidate by make_mixed_omp.py.
// The production FP64 residual, ordered dots and least-squares solve are kept.
#include "cpu/wnnm/jacobi8.hpp"
#include "nss/cpu_common.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace nss::detail {
struct MixedOmpStats {
    unsigned attempts = 0;
    unsigned accepted = 0;
    unsigned refined_columns = 0;
};

class MixedOmpSearch {
public:
    MixedOmpSearch(const float* dictionary, int rows, int atoms, int stride)
        : dictionary_(dictionary), rows_(rows), atoms_(atoms), stride_(stride),
          enabled_(rows >= 16 && rows <= 256 && atoms >= 16 && atoms <= 256) {
        if (!enabled_) return;
        float maximum = 0;
        for (int atom = 0; atom < atoms; ++atom) {
            const float* column = dictionary + static_cast<std::size_t>(atom) * stride;
            for (int row = 0; row < rows; ++row)
                maximum = std::max(maximum, std::abs(column[row]));
        }
        maximum_ = maximum;
        enabled_ = maximum > 0 && is_finite_bits(maximum);
    }

    bool select(const float* residual_storage, const unsigned char* used,
                int& selected, double& magnitude, MixedOmpStats* stats = nullptr) {
        if (!enabled_) return false;
        if (stats) ++stats->attempts;
        double sum_abs = 0, cast_error = 0;
        for (int row = 0; row < rows_; ++row) {
            double value;
            std::memcpy(&value, residual_storage + 2 * static_cast<std::size_t>(row), sizeof(value));
            if (!(std::abs(value) <= double(std::numeric_limits<float>::max()))) return false;
            const float rounded = static_cast<float>(value);
            if (!is_finite_bits(rounded)) return false;
            residual_[row] = rounded;
            sum_abs += std::abs(double(rounded));
            cast_error += std::abs(value - double(rounded));
        }
        // Bound conversion + dot/reduction roundoff for any supported lane
        // width. The positive allowance also covers flushed FP32 operands
        // and subnormal results. Factor two safely encloses bound evaluation
        // and the much smaller ordered-FP64 reference dot roundoff.
        constexpr double unit = 0x1p-24;
        const double operations = 2.0 * rows_ + 32;
        const double gamma = operations * unit / (1 - operations * unit);
        const double tiny = double(std::numeric_limits<float>::min());
        const double error = 2 * maximum_ * (cast_error + gamma * sum_abs) +
            (2.0 * rows_ + 64) * tiny * (1 + sum_abs + rows_ * maximum_);
        if (!std::isfinite(error)) return false;
        gemm_tn_hwy(rows_, 1, atoms_, dictionary_, stride_, residual_, rows_, approximate_, atoms_);
        int seed = -1;
        float peak = 0;
        for (int atom = 0; atom < atoms_; ++atom) {
            if (!is_finite_bits(approximate_[atom])) return false;
            if (!used[atom] && std::abs(approximate_[atom]) > peak) {
                peak = std::abs(approximate_[atom]);
                seed = atom;
            }
        }
        if (seed < 0 || double(peak) - error <= double(1e-12f)) return false;
        auto precise = [&](int atom) {
            const float* column = dictionary_ + static_cast<std::size_t>(atom) * stride_;
            double dot = 0;
            for (int row = 0; row < rows_; ++row) {
                double value;
                std::memcpy(&value, residual_storage + 2 * static_cast<std::size_t>(row), sizeof(value));
                dot = std::fma(double(column[row]), value, dot);
            }
            return std::abs(dot);
        };
        double best_value = precise(seed);
        if (!std::isfinite(best_value)) return false;
        int contenders = 0;
        for (int atom = 0; atom < atoms_; ++atom)
            if (!used[atom] && double(std::abs(approximate_[atom])) + error >= best_value) ++contenders;
        // Dense ambiguity is cheaper in the original eight-dot FP64 loop.
        if (contenders > 16) return false;
        int best = seed;
        unsigned refined = 1;
        for (int atom = 0; atom < atoms_; ++atom) {
            if (used[atom] || atom == seed || double(std::abs(approximate_[atom])) + error < best_value) continue;
            const double value = precise(atom);
            ++refined;
            if (!std::isfinite(value)) return false;
            if (value > best_value || (value == best_value && atom < best)) {
                best = atom;
                best_value = value;
            }
        }
        selected = best;
        magnitude = best_value;
        if (stats) { ++stats->accepted; stats->refined_columns += refined; }
        return true;
    }

private:
    const float* dictionary_;
    int rows_, atoms_, stride_;
    bool enabled_;
    double maximum_ = 0;
    float residual_[256], approximate_[256];
};
}  // namespace nss::detail
