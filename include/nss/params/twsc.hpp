// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved TWSC arguments shared by every backend.
#include "nss/params.hpp"

namespace nss {

struct TwscSolverOptions {
    int iterations = 10;
    double rho = 0.5;
    double mu = 1.1;
    double tolerance = 1e-6;
};

struct TwscImageOptions {
    int block = kTwscDefaultBlock, group = kTwscDefaultGroup, iterations = kTwscDefaultIters;
    int step = kTwscDefaultStep, window = 60, radius = 0, ps_num = 2, ps_range = 4;
    double lambda2 = 1, delta = 0;
    TwscSolverOptions solver;
};

}  // namespace nss
