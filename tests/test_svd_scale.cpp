#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numbers>
#include <vector>

namespace {
constexpr float guard = -123456.f;
double basis(int row, int col, int rows) {
    return std::cos(std::numbers::pi * (row + .5) * col / rows) * std::sqrt((col ? 2. : 1.) / rows);
}

bool check(int m, int n, int rank, int exponent, int count, bool leading_zero) {
    const int r = std::min(m, n), lda = m + 3, ldu = m + 5, ldvt = n + 3;
    const double scale = std::ldexp(1., exponent);
    std::vector<float> a(lda * n, guard);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double value = 0.;
            if (leading_zero) {
                value = col > 0 && row == col - 1 ? 1. : 0.;
            } else {
                for (int k = 0; k < (rank < 0 ? r : rank); ++k)
                    value += basis(row, k, m) * basis(col, k, n) * (rank < 0 ? std::pow(.125, k) : 1.);
            }
            a[row + col * lda] = static_cast<float>(value * scale);
        }
    }
    std::vector<std::vector<float>> u(count), s(count), vt(count), work(count);
    std::vector<nss::SvdBatchItem> items(count);
    std::vector<int> statuses(count, 0);
    for (int b = 0; b < count; ++b) {
        u[b].assign(ldu * n, guard);s[b].assign(r + 1, guard);vt[b].assign(ldvt * n, guard);
        work[b].resize(nss::wnnm_shrink_work_floats(m, n));
        items[b] = {m,n,a.data(),lda,u[b].data(),ldu,s[b].data(),vt[b].data(),ldvt,
                    work[b].data(),static_cast<int>(work[b].size()),&statuses[b]};
    }
    const int rc = count == 1 ? nss::svd_economy(m,n,a.data(),lda,u[0].data(),ldu,s[0].data(),vt[0].data(),ldvt) :
                               nss::svd_economy_batch(items.data(),count);
    if (rc != 0) { std::fprintf(stderr,"SVD failed m=%d n=%d rank=%d exp=%d count=%d rc=%d\n",m,n,rank,exponent,count,rc); return false; }
    for (int b = 0; b < count; ++b) {
        if (count > 1 && statuses[b] != 1) return false;
        double energy = 0., error = 0., pca_error = 0., orth_u = 0., orth_v = 0., spectrum = 0.;
        for (int k = 0; k < r; ++k) {
            const double expected = rank < 0 ? std::pow(.125, k) : k < rank ? 1. : 0.;
            spectrum = std::max(spectrum, std::fabs(s[b][k] / scale - expected));
            if (!std::isfinite(s[b][k])) return false;
            for (int q = 0; q < r; ++q) {
                if (s[b][k] / scale <= 1e-5 || s[b][q] / scale <= 1e-5) continue;
                double du = 0., dv = 0.;
                for (int row = 0; row < m; ++row) du += static_cast<double>(u[b][row+k*ldu])*u[b][row+q*ldu];
                for (int col = 0; col < n; ++col) dv += static_cast<double>(vt[b][k+col*ldvt])*vt[b][q+col*ldvt];
                orth_u = std::max(orth_u,std::fabs(du-(k==q)));
                orth_v = std::max(orth_v,std::fabs(dv-(k==q)));
            }
        }
        for (int col = 0; col < n; ++col) {
            std::vector<double> projection(r);
            for (int k = 0; k < r; ++k) {
                for (int row = 0; row < m; ++row) projection[k] += u[b][row+k*ldu]*(a[row+col*lda]/scale);
            }
            for (int row = 0; row < m; ++row) {
                const double want = a[row+col*lda]/scale;
                double rec = 0., pca = 0.;
                for (int k = 0; k < r; ++k) {
                    rec += u[b][row+k*ldu]*(s[b][k]/scale)*vt[b][k+col*ldvt];
                    pca += u[b][row+k*ldu]*projection[k];
                }
                energy += want*want;error += (rec-want)*(rec-want);pca_error += (pca-want)*(pca-want);
            }
            for (int row=m;row<ldu;++row) if(u[b][row+col*ldu]!=guard) return false;
            for (int row=n;row<ldvt;++row) if(vt[b][row+col*ldvt]!=guard) return false;
        }
        if(s[b].back()!=guard) return false;
        error=std::sqrt(error/std::max(energy,1.));pca_error=std::sqrt(pca_error/std::max(energy,1.));
        if (!(error<=5e-5 && pca_error<=5e-5 && spectrum<=5e-5 && orth_u<=5e-4 && orth_v<=5e-4)) {
            std::fprintf(stderr,"scale m=%d n=%d rank=%d exp=%d count=%d leading0=%d: rec=%g PCA=%g S=%g U=%g V=%g\n",
                         m,n,rank,exponent,count,leading_zero,error,pca_error,spectrum,orth_u,orth_v);
            return false;
        }
    }
    return true;
}
}

int main() {
    int cases=0;
    for(int n:{1,2,3,4,7,8,9,16,31,32}) {
        for(int m:{n,std::max(1,n/2),2*n,256}) {
            for(int rank:{0,1,std::min(m,n),-1}) {
                for(int exponent:{-100,-32,-16,0,16,64,100}) {
                    for(int count:{1,5}) {
                        if(!check(m,n,rank,exponent,count,false)) return 1;
                        ++cases;
                    }
                }
            }
        }
    }
    for(int n:{4,8,16,32}) {
        for(int exponent:{-32,0,32}) {
            if(!check(2*n,n,n-1,exponent,1,true)) return 1;
            if(!check(2*n,n,n-1,exponent,5,true)) return 1;
        }
    }
    std::printf("SVD scale/rank/PCA/stride checks passed: %d fixtures plus leading-zero controls\n",cases);
    return 0;
}
