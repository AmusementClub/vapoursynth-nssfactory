#include "nss/cpu_twsc_full.hpp"
#include "cpu/hwy_config.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/twsc/svd_batch.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

bool TwscSvdValidationLanesAvailable() {
    // Admission is currently the measured AVX2 lane. Other targets retain the
    // scalar validator until their own complete-pipeline timing is available.
    return HWY_TARGET == HWY_AVX2;
}

bool TwscValidSvdLanes(const float* a, int m, int n, int lda, const TwscWorkspace& w) {
    const hn::ScalableTag<double> d;
    const hn::RebindToSigned<decltype(d)> di;
    const int lanes = int(hn::Lanes(d)), r = std::min(m, n);
    HWY_ALIGN double first[HWY_MAX_BYTES / sizeof(double)];
    HWY_ALIGN double second[HWY_MAX_BYTES / sizeof(double)];
    HWY_ALIGN std::int64_t offsets[HWY_MAX_BYTES / sizeof(double)];
    for (int lane = 0; lane < lanes; ++lane) offsets[lane] = std::int64_t(lane) * m;
    const auto column_offsets = hn::LoadU(di, offsets);
    double error = 0, norm = 0, orthogonal = 0;
    for (int j = 0; j < n; ++j) {
        int i = 0;
        for (; i + lanes <= m; i += lanes) {
            auto x = hn::Zero(d);
            for (int k = 0; k < r; ++k) {
                const auto atom = hn::Mul(hn::LoadU(d, w.dictionary.data() + i + k * m), hn::Set(d, w.singular[k]));
                x = hn::Add(x, hn::Mul(atom, hn::Set(d, w.vt[k + j * r])));
            }
            hn::StoreU(x, d, first);
            // Keep the original j/i accumulation and its finite-value checks.
            for (int lane = 0; lane < lanes; ++lane) {
                if (!std::isfinite(first[lane])) return false;
                const double y = a[i + lane + j * lda], delta = first[lane] - y;
                error += delta * delta; norm += y * y;
            }
        }
        for (; i < m; ++i) {
            double x = 0;
            for (int k = 0; k < r; ++k) x += w.dictionary[i + k * m] * w.singular[k] * w.vt[k + j * r];
            if (!std::isfinite(x)) return false;
            const double y = a[i + j * lda], delta = x - y;
            error += delta * delta; norm += y * y;
        }
    }
    for (int j = 0; j < r; ++j) if (w.singular[j] > 0) {
        int k = 0;
        for (; k + lanes <= j + 1; k += lanes) {
            auto u = hn::Zero(d), v = hn::Zero(d);
            for (int i = 0; i < m; ++i)
                u = hn::Add(u, hn::Mul(hn::Set(d, w.dictionary[i + j * m]),
                                      hn::GatherIndex(d, w.dictionary.data() + i + k * m, column_offsets)));
            for (int i = 0; i < n; ++i)
                v = hn::Add(v, hn::Mul(hn::Set(d, w.vt[j + i * r]), hn::LoadU(d, w.vt.data() + k + i * r)));
            hn::StoreU(u, d, first); hn::StoreU(v, d, second);
            for (int lane = 0; lane < lanes; ++lane) if (w.singular[k + lane] > 0) {
                const double diagonal = j == k + lane ? 1.0 : 0.0;
                orthogonal = std::max(orthogonal, std::abs(first[lane] - diagonal));
                orthogonal = std::max(orthogonal, std::abs(second[lane] - diagonal));
            }
        }
        for (; k <= j; ++k) if (w.singular[k] > 0) {
            double dot = 0;
            for (int i = 0; i < m; ++i) dot += w.dictionary[i + j * m] * w.dictionary[i + k * m];
            orthogonal = std::max(orthogonal, std::abs(dot - (j == k ? 1.0 : 0.0)));
            dot = 0;
            for (int i = 0; i < n; ++i) dot += w.vt[j + i * r] * w.vt[k + i * r];
            orthogonal = std::max(orthogonal, std::abs(dot - (j == k ? 1.0 : 0.0)));
        }
    }
    return error <= norm * 2.5e-9 && orthogonal <= 5e-4;
}

// Lanes are independent groups. Every per-group dot product and rotation has
// the scalar FP64 operation order; there is no horizontal reduction or FMA.
void TwscSvd64Batch(TwscWorkspace* const* work, int m, int n, int count) {
    const hn::ScalableTag<double> d;
    const int lanes = int(hn::Lanes(d)), rows = std::max(m,n), r = std::min(m,n);
    const auto zero = hn::Zero(d), one = hn::Set(d,1.0);
    const bool transposed = m < n;
    ResourceVector<double> storage(checked_product({std::size_t(lanes), std::size_t(2*rows*r+2*r*r+2*r)}));
    double* qr=storage.data(); double* b=qr+rows*r*lanes;
    double* v=b+r*r*lanes; double* u=v+r*r*lanes;
    double* tau=u+rows*r*lanes; double* singular=tau+r*lanes;
    auto get=[&](const double* a,int i){ return hn::LoadU(d,a+std::size_t(i)*lanes); };
    auto put=[&](double* a,int i,auto x){ hn::StoreU(x,d,a+std::size_t(i)*lanes); };
    ResourceVector<double> scales(lanes);
    for (int start=0; start<count; start+=lanes) {
        const int real=std::min(lanes,count-start);
        // qr is fully overwritten by the pack loop below; only b (lower
        // triangle), v (off-diagonal), u, tau and singular need the zero fill.
        std::fill(b,storage.data()+storage.size(),0.0);
        for (int lane=0; lane<lanes; ++lane) {
            auto& w=*work[start+std::min(lane,real-1)];
            double maximum=0;
            for (float x:w.input) maximum=std::max(maximum,std::abs(double(x)));
            int exponent=0; if(maximum>0) std::frexp(maximum,&exponent);
            const double scale=std::ldexp(1.0,-exponent);scales[lane]=scale;
            for(int j=0;j<r;++j) for(int i=0;i<rows;++i)
                qr[(i+j*rows)*lanes+lane]=double(w.input[transposed?j+i*m:i+j*m])*scale;
        }
        for(int k=0;k<r;++k) {
            auto norm=zero;
            for(int i=k;i<rows;++i) { const auto x=get(qr,i+k*rows);norm=hn::Add(norm,hn::Mul(x,x)); }
            const auto active=hn::Gt(norm,zero);
            const auto x=get(qr,k+k*rows);
            const auto alpha=hn::Neg(hn::CopySign(hn::Sqrt(norm),x));
            const auto inv=hn::Div(one,hn::IfThenElse(active,hn::Sub(x,alpha),one));
            const auto tk=hn::IfThenElse(active,hn::Div(hn::Sub(alpha,x),hn::IfThenElse(active,alpha,one)),zero);
            put(tau,k,tk);
            for(int i=k+1;i<rows;++i) put(qr,i+k*rows,hn::Mul(get(qr,i+k*rows),inv));
            put(qr,k+k*rows,hn::IfThenElse(active,alpha,x));
            for(int j=k+1;j<r;++j) {
                auto dot=get(qr,k+j*rows);
                for(int i=k+1;i<rows;++i) dot=hn::Add(dot,hn::Mul(get(qr,i+k*rows),get(qr,i+j*rows)));
                dot=hn::Mul(dot,tk);
                put(qr,k+j*rows,hn::Sub(get(qr,k+j*rows),dot));
                for(int i=k+1;i<rows;++i) put(qr,i+j*rows,hn::Sub(get(qr,i+j*rows),hn::Mul(get(qr,i+k*rows),dot)));
            }
        }
        auto initial=zero;
        for(int j=0;j<r;++j) {
            put(v,j+j*r,one);auto norm=zero;
            for(int i=0;i<=j;++i) { const auto x=get(qr,i+j*rows);put(b,i+j*r,x);norm=hn::Add(norm,hn::Mul(x,x)); }
            initial=hn::Max(initial,norm);
        }
        const auto floor=hn::Mul(initial,hn::Set(d,1e-14));
        const auto tolerance=hn::Set(d,1e-6);
        for(int sweep=0;sweep<128;++sweep) {
            bool rotated=false;
            for(int p=0;p<r-1;++p) for(int q=p+1;q<r;++q) {
                auto app=zero,aqq=zero,apq=zero;
                for(int i=0;i<r;++i) {
                    const auto x=get(b,i+p*r),y=get(b,i+q*r);
                    app=hn::Add(app,hn::Mul(x,x));aqq=hn::Add(aqq,hn::Mul(y,y));apq=hn::Add(apq,hn::Mul(x,y));
                }
                const auto active=hn::And(hn::And(hn::Gt(app,floor),hn::Gt(aqq,floor)),
                    hn::Gt(hn::Abs(apq),hn::Mul(tolerance,hn::Sqrt(hn::Mul(app,aqq)))));
                if(hn::AllFalse(d,active)) continue;
                rotated=true;
                const auto z=hn::Div(hn::Sub(aqq,app),hn::IfThenElse(active,hn::Mul(hn::Set(d,2),apq),one));
                const auto t=hn::Div(hn::CopySign(one,z),hn::Add(hn::Abs(z),hn::Sqrt(hn::Add(one,hn::Mul(z,z)))));
                const auto c=hn::Div(one,hn::Sqrt(hn::Add(one,hn::Mul(t,t)))),s=hn::Mul(t,c);
                for(int i=0;i<r;++i) {
                    const auto x=get(b,i+p*r),y=get(b,i+q*r);
                    put(b,i+p*r,hn::IfThenElse(active,hn::Sub(hn::Mul(c,x),hn::Mul(s,y)),x));
                    put(b,i+q*r,hn::IfThenElse(active,hn::Add(hn::Mul(s,x),hn::Mul(c,y)),y));
                    const auto vx=get(v,i+p*r),vy=get(v,i+q*r);
                    put(v,i+p*r,hn::IfThenElse(active,hn::Sub(hn::Mul(c,vx),hn::Mul(s,vy)),vx));
                    put(v,i+q*r,hn::IfThenElse(active,hn::Add(hn::Mul(s,vx),hn::Mul(c,vy)),vy));
                }
            }
            if(!rotated) break;
        }
        for(int j=0;j<r;++j) {
            auto norm=zero;
            for(int i=0;i<r;++i) {const auto x=get(b,i+j*r);norm=hn::Add(norm,hn::Mul(x,x));}
            put(singular,j,hn::IfThenElse(hn::Gt(norm,floor),hn::Sqrt(norm),zero));
        }
        // Match scalar stable max-selection exactly, including equal spectra.
        for(int lane=0;lane<lanes;++lane) for(int j=0;j<r;++j) {
            int best=j;
            for(int k=j+1;k<r;++k) if(singular[k*lanes+lane]>singular[best*lanes+lane]) best=k;
            if(best==j) continue;
            std::swap(singular[j*lanes+lane],singular[best*lanes+lane]);
            for(int i=0;i<r;++i) {
                std::swap(b[(i+j*r)*lanes+lane],b[(i+best*r)*lanes+lane]);
                std::swap(v[(i+j*r)*lanes+lane],v[(i+best*r)*lanes+lane]);
            }
        }
        for(int j=0;j<r;++j) {
            const auto sj=get(singular,j);
            const auto active=hn::Gt(sj,zero);
            const auto den=hn::IfThenElse(active,sj,one);
            for(int i=0;i<r;++i) put(u,i+j*rows,hn::IfThenElse(active,hn::Div(get(b,i+j*r),den),zero));
        }
        for(int k=r-1;k>=0;--k) {
            const auto tk=get(tau,k);
            for(int j=0;j<r;++j) {
                auto dot=get(u,k+j*rows);
                for(int i=k+1;i<rows;++i) dot=hn::Add(dot,hn::Mul(get(qr,i+k*rows),get(u,i+j*rows)));
                dot=hn::Mul(dot,tk);
                put(u,k+j*rows,hn::Sub(get(u,k+j*rows),dot));
                for(int i=k+1;i<rows;++i) put(u,i+j*rows,hn::Sub(get(u,i+j*rows),hn::Mul(get(qr,i+k*rows),dot)));
            }
        }
        for(int lane=0;lane<real;++lane) {
            auto& w=*work[start+lane];
            w.dictionary.resize(std::size_t(m)*r);w.singular.resize(r);w.vt.resize(std::size_t(r)*n);
            for(int k=0;k<r;++k) {
                const double sk=singular[k*lanes+lane];w.singular[k]=sk/scales[lane];
                for(int i=0;i<m;++i) w.dictionary[i+k*m]=sk==0?0:(transposed?v[(i+k*r)*lanes+lane]:u[(i+k*rows)*lanes+lane]);
                for(int j=0;j<n;++j) w.vt[k+j*r]=sk==0?0:(transposed?u[(j+k*rows)*lanes+lane]:v[(j+k*r)*lanes+lane]);
            }
        }
    }
}
} // namespace HWY_NAMESPACE
} // namespace nss
HWY_AFTER_NAMESPACE();
#if HWY_ONCE
namespace nss {
HWY_EXPORT(TwscSvd64Batch);
HWY_EXPORT(TwscValidSvdLanes);
HWY_EXPORT(TwscSvdValidationLanesAvailable);
bool twsc_svd_validation_lanes_available() {
    return HWY_DYNAMIC_DISPATCH(TwscSvdValidationLanesAvailable)();
}
bool twsc_valid_svd_lanes(const float* a, int m, int n, int lda, const TwscWorkspace& work) {
    return HWY_DYNAMIC_DISPATCH(TwscValidSvdLanes)(a, m, n, lda, work);
}
void twsc_svd64_batch(TwscWorkspace* const* work,int m,int n,int count) {
    if(!work || m<1 || m>kTwscMaxRows || n<1 || n>kTwscMaxColumns || count<1 || count>16)
        throw std::invalid_argument("nss.TWSC: invalid full SVD batch");
    for(int i=0;i<count;++i) if(!work[i] || work[i]->input.size()!=std::size_t(m)*n)
        throw std::invalid_argument("nss.TWSC: invalid prepared SVD batch");
    HWY_DYNAMIC_DISPATCH(TwscSvd64Batch)(work,m,n,count);
}
} // namespace nss
#endif
