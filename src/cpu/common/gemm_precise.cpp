#include "nss/cpu_linalg.hpp"
#include "cpu/hwy_config.hpp"
#include <algorithm>
#include <cstddef>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/common/gemm_precise.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn=hwy::HWY_NAMESPACE;

void GemmF32F64(const float* a,int lda,const float* b,int ldb,float* c,int ldc,int m,int n,int k) {
    const hn::ScalableTag<double> dd;
    const hn::Rebind<float,decltype(dd)> df;
    const int lanes=static_cast<int>(hn::Lanes(dd));
    for(int row=0;row<m;++row)for(int col=0;col<n;col+=lanes) {
        auto sum=hn::Zero(dd);
        const auto valid=static_cast<std::size_t>(std::min(lanes,n-col));
        for(int inner=0;inner<k;++inner) {
            const auto av=hn::Set(df,a[static_cast<std::size_t>(row)*lda+inner]);
            const auto bv=hn::LoadN(df,b+static_cast<std::size_t>(inner)*ldb+col,valid);
            sum=hn::Add(sum,hn::PromoteTo(dd,hn::Mul(av,bv)));
        }
        hn::StoreN(hn::DemoteTo(df,sum),df,c+static_cast<std::size_t>(row)*ldc+col,valid);
    }
}
}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(GemmF32F64);
int gemm_f32_f64(const float* a,int lda,const float* b,int ldb,float* c,int ldc,int m,int n,int k) {
    if(m<0 || n<0 || k<0 || lda<k || ldb<n || ldc<n) return -1;
    if(m==0 || n==0) return 0;
    if(!c || (k && (!a || !b))) return -1;
    HWY_DYNAMIC_DISPATCH(GemmF32F64)(a,lda,b,ldb,c,ldc,m,n,k);
    return 0;
}
}
#endif
