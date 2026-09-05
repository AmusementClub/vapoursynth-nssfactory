#pragma once

namespace nss {

inline constexpr int kNlmDefaultD = 1;
inline constexpr int kNlmDefaultA = 2;
inline constexpr int kNlmDefaultS = 4;
inline constexpr float kNlmDefaultH = 1.2f;
inline constexpr float kNlmDefaultWref = 1.0f;

inline constexpr int kBmBlock = 8;
inline constexpr int kBmGroup = 8;
inline constexpr int kBmMaxBlock = 32;
inline constexpr int kBmMaxGroup = 64;
inline constexpr int kBmDefaultStep = 8;
inline constexpr int kBmDefaultRange = 7;
inline constexpr int kBmMaxRange = 64;
inline constexpr int kBmDefaultPsNum = 2;
inline constexpr int kBmDefaultPsRange = 4;
inline constexpr int kBmMaxRadius = 16;
inline constexpr float kBmDefaultSigma = 3.0f;
inline constexpr float kBmHardLambda = 2.7f;

inline bool bm_allowed_block(int v) {
    return v == 1 || v == 2 || v == 4 || v == 8 || v == 12 || v == 16 || v == 32;
}

inline bool lssc_allowed_block(int v) {
    return v == 1 || v == 2 || v == 4 || v == 8 || v == 16;
}

inline bool bm_allowed_group(int v) {
    return v == 1 || v == 2 || v == 4 || v == 8 || v == 16 || v == 32 || v == 64;
}

inline constexpr int kWnnmMaxBlock = 16;
inline constexpr int kWnnmMaxGroup = 32;
inline constexpr int kWnnmDefaultBlock = 8;
inline constexpr int kWnnmDefaultStep = 8;
inline constexpr int kWnnmDefaultGroup = 8;
inline constexpr int kWnnmDefaultRange = 7;
inline constexpr int kWnnmDefaultRadius = 0;
inline constexpr int kWnnmDefaultPsNum = 2;
inline constexpr int kWnnmDefaultPsRange = 4;
inline constexpr int kWnnmDefaultResidual = 0;
// residual=1 demeans rows (Matlab Estimation pipeline). residual=0 keeps DC in
// the ADMM matrix (bare MCWNNM_ADMM.m). Factory default matches Estimation.
inline constexpr int kMcwnnmDefaultResidual = 1;
inline constexpr int kWnnmDefaultAdaptive = 1;
inline constexpr float kWnnmDefaultSigma = 3.0f;

inline constexpr int kSvdMaxM = 256;
inline constexpr int kSvdMaxN = 32;

inline constexpr int kMcwnnmDefaultBlock = 8;
inline constexpr int kMcwnnmDefaultStep = 8;
inline constexpr int kMcwnnmDefaultGroup = 8;
inline constexpr int kMcwnnmDefaultRange = 7;
inline constexpr int kMcwnnmDefaultAdmmIter = 10;
inline constexpr float kMcwnnmDefaultRho = 3.f;
inline constexpr float kMcwnnmDefaultMu = 1.001f;
inline constexpr int kMcwnnmDefaultAdaptive = 0;
inline constexpr float kMcwnnmDefaultSigma = 3.0f;
inline constexpr int kMcwnnmDefaultIters = 2;
inline constexpr float kMcwnnmDefaultDelta = 0.1f;

inline constexpr int kTwscDefaultBlock = 8;
inline constexpr int kTwscDefaultStep = 8;
inline constexpr int kTwscDefaultGroup = 8;
inline constexpr int kTwscDefaultRange = 7;
inline constexpr float kTwscDefaultLambda1 = 0.f;
inline constexpr float kTwscDefaultLambda2 = 3.f;
inline constexpr float kTwscDefaultSigma = 3.0f;
inline constexpr int kTwscDefaultIters = 2;
inline constexpr float kTwscDefaultDelta = 0.1f;

inline constexpr int kNlhDefaultBlock = 8;
inline constexpr int kNlhDefaultStep = 8;
inline constexpr int kNlhDefaultGroup = 16;
inline constexpr int kNlhDefaultRange = 20;
inline constexpr int kNlhDefaultQ = 4;
inline constexpr float kNlhDefaultSigma = 3.0f;

inline constexpr int kNcsrDefaultBlock = 8;
inline constexpr int kNcsrDefaultStep = 8;
inline constexpr int kNcsrDefaultGroup = 8;
inline constexpr int kNcsrDefaultRange = 7;
inline constexpr int kNcsrDefaultIters = 2;
inline constexpr float kNcsrDefaultDelta = 0.1f;
inline constexpr float kNcsrDefaultSigma = 3.0f;

inline constexpr int kLsscDefaultBlock = 8;
inline constexpr int kLsscDefaultStep = 8;
inline constexpr int kLsscDefaultAtoms = 256;
inline constexpr int kLsscDefaultClusters = 64;
inline constexpr float kLsscDefaultSigma = 3.0f;

enum class Distance { SSD };

struct SearchConfig {
    int block = kBmBlock;
    int step = kBmDefaultStep;
    int group = kBmGroup;
    int bm_range = kBmDefaultRange;
    int radius = 0;
    int ps_num = kBmDefaultPsNum;
    int ps_range = kBmDefaultPsRange;
};

}  // namespace nss
