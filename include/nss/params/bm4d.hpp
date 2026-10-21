// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Resolved BM4D arguments, constants and volume geometry shared by every backend.
//
// BM4D (Maggioni, Katkovnik, Egiazarian and Foi, "Nonlocal Transform-Domain
// Filter for Volumetric Data Denoising and Reconstruction", IEEE TIP 22(1),
// 2013), model version 1, written from the paper with the authors' published
// default parameters (their v4 profile "np"); none of their software is used.
// A clip is a volume with the frames as its third dimension. Each plane is
// filtered on its own.
//
// Volumes: the frames are cut into chunks of kBm4dChunk; a chunk is filtered
// as the volume of its frames plus up to kBm4dChunkPad frames on each side
// (within the clip), and each plane into tiles (bm4d_tiles) filtered as the
// tile plus up to kBm4dTilePad pixels on each side (within the plane). Only
// the chunk's frames and the tile's pixels are kept. The geometry is part of
// the model.
//
// Each stage, on a volume: reference cubes of edge L at every step-th
// position along each axis (and the last one); for each, the cubes within
// +-kBm4dSearch along every axis whose squared distance to it, divided by
// L^3 sigma^2, is at most tau, the closest M (the reference first; ties keep
// the earlier offset in (dz, dy, dx) scan order), and of those the largest
// power of two; every cube through the orthonormal 3D DCT, the group through
// the orthonormal Haar transform;
//   hard stage (on the noisy volume): coefficients below lambda sigma in
//   magnitude set to zero, weight 1 / (sigma^2 N) with N the coefficients
//   kept (at least 1);
//   Wiener stage (matched on the basic estimate): the noisy group times
//   W = B^2 / (B^2 + sigma^2), B the basic group's coefficients, weight
//   1 / (sigma^2 sum W^2);
// then back, and added to numerator and denominator with the weight times
// a Kaiser window (beta kBm4dKaiserBeta) in group order, references in
// (z, y, x) raster order. The estimate is numerator / denominator.
// Distances are sums in a fixed order (along z, then y, then x, each in
// increasing order), so the output does not depend on the vector width.
#include "nss/params.hpp"

namespace nss {

struct Bm4dStage {
    int cube;        // L
    int step;
    int group;       // M
    float tau;       // matching threshold, in sigma^2 per voxel
};

inline constexpr Bm4dStage kBm4dHard{4, 3, 16, 2.9527f};
inline constexpr Bm4dStage kBm4dWiener{5, 3, 32, 0.7689f};
inline constexpr int kBm4dSearch = 7;
inline constexpr float kBm4dLambda = 2.7f;
inline constexpr float kBm4dKaiserBeta = 2.f;
inline constexpr int kBm4dChunk = 32;
inline constexpr int kBm4dChunkPad = 8;
inline constexpr int kBm4dTile = 256;
inline constexpr int kBm4dTilePad = 16;
inline constexpr float kBm4dDefaultSigma = 3.f;

struct Bm4dParams {
    float sigma[3]{kBm4dDefaultSigma, kBm4dDefaultSigma, kBm4dDefaultSigma};  // 8-bit units, true noise
};

// Tiles along an edge of n pixels: n / kBm4dTile rounded to nearest, at
// least one; tile i covers [n i / count, n (i + 1) / count).
inline int bm4d_tile_count(int n) {
    const int count = (n + kBm4dTile / 2) / kBm4dTile;
    return count > 0 ? count : 1;
}
inline int bm4d_tile_start(int n, int count, int i) {
    return static_cast<int>(static_cast<long long>(n) * i / count);
}
inline int bm4d_tile_max(int n) {
    const int count = bm4d_tile_count(n);
    int best = 0;
    for (int i = 0; i < count; ++i) {
        const int e = bm4d_tile_start(n, count, i + 1) - bm4d_tile_start(n, count, i);
        best = e > best ? e : best;
    }
    return best;
}
// The chunk holding frame n: [first, first + kBm4dChunk) within the clip.
inline int bm4d_chunk_first(int n) { return n - n % kBm4dChunk; }

}  // namespace nss
