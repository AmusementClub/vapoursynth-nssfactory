// SPDX-License-Identifier: GPL-2.0-only
#pragma once
#include <VapourSynth4.h>
#include "nss/contracts.hpp"
#include "nss/params/full_image.hpp"
#include "frontend/temporal_args.hpp"

namespace nss::frontend {

inline constexpr const char* kTwscSignature =
    "clip:vnode;sigma:float[]:opt;block_size:int:opt;block_step:int:opt;group_size:int:opt;"
    "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;lambda2:float:opt;"
    "rclip:vnode:opt;iters:int:opt;delta:float:opt;estimate_sigma:int:opt;search_window:int:opt;"
    "admm_iter:int:opt;rho:float:opt;mu:float:opt;tol:float:opt;memory_limit_mb:int:opt;" NSS_TEMPORAL_SIGNATURE;

inline constexpr const char* kNlhSignature =
    "clip:vnode;sigma:float[]:opt;block_size:int[]:opt;block_step:int[]:opt;group_size:int[]:opt;"
    "bm_range:int:opt;radius:int:opt;ps_num:int:opt;ps_range:int:opt;q:int[]:opt;rclip:vnode:opt;"
    "noise_model:data:opt;search_window:int[]:opt;basic_iters:int:opt;lambda_basic:float:opt;"
    "hard_strength:float:opt;wiener_iters:int:opt;wiener_sigma_scale:float:opt;memory_limit_mb:int:opt;" NSS_TEMPORAL_SIGNATURE;

// Arguments of TWSC (model == Model::TWSC) or NLH. rclip_vi is null when no
// rclip was given.
FullImageParams parse_full_image(const VSAPI* api, const VSMap* in, const VSVideoInfo& vi,
                                 const VSVideoInfo* rclip_vi, Model model, const char* ns);

// Creation-time geometry: the fixed 8x8 bootstrap for blind estimation, or
// the resolved block sizes/steps against every selected plane.
void validate_full_image_geometry(const FullImageParams& p, const VSVideoInfo& vi, Model model, const char* ns);

}  // namespace nss::frontend
