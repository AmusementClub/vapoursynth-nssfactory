#pragma once
#include "nss/cpu_batch.hpp"
#include <functional>

// Version-2 trace transport is retained only for frozen historical sources.
// Version-3 TWSC uses alignment_probe and its full-objective FP64 oracle.
#if __has_include("nss/cpu_twsc_full.hpp")
struct LegacyTwscTraceItem {
    float* group = nullptr; int m = 0, n = 0, lda = 0; float sigma = 0;
    float* work = nullptr; int work_floats = 0; const float* col_sigma = nullptr;
    float* col_weight = nullptr; const float* row_weight = nullptr; int* status = nullptr;
};
#else
using LegacyTwscTraceItem = nss::TwscPcaBatchItem;
#endif
int nss_trace_bm_groups(nss::Bm3dFilterBatchItem*, int, int (*)(nss::Bm3dFilterBatchItem*, int));
int nss_trace_ncsr_groups(nss::NcsrFilterBatchItem*, int, int (*)(nss::NcsrFilterBatchItem*, int));
int nss_trace_mc_groups(nss::McwnnmFilterBatchItem*, int, int (*)(nss::McwnnmFilterBatchItem*, int));
int nss_trace_twsc_groups(LegacyTwscTraceItem*, int, int (*)(LegacyTwscTraceItem*, int));
void nss_trace_bm_call(float*, int, int, int, int, float, bool, const float*, float*, float*, const std::function<void()>&);
int nss_trace_twsc_call(float*, int, int, int, float, float*, int, const float*, float*, const float*, const std::function<int()>&);
