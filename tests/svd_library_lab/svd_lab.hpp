#pragma once
#include "nss/cpu_twsc_full.hpp"
namespace nss {
int svd_lab_mode(); // 0=current, 1=tighter FP64, 2=DGESDD, 3=DGESVD
bool svd_lab_decompose(const float*,int,int,int,TwscWorkspace&);
void svd_lab_fallback();
void svd_lab_diagonal(bool);
void svd_lab_capture(const float*,int,int);
}
