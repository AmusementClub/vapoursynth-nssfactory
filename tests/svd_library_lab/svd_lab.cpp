// SPDX-License-Identifier: GPL-2.0-only
#include "cpu/twsc/svd_lab.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <stdexcept>
extern "C" {
void dgesdd_(const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*,int*);
void dgesvd_(const char*,const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*);
int openblas_get_num_threads();
char* openblas_get_config();
}
namespace nss {
int svd_lab_mode() {
    static int mode=[] {const char* s=std::getenv("NSS_SVD_LAB");
        if(!s || !std::strcmp(s,"baseline"))return 0;
        if(!std::strcmp(s,"tight"))return 1;
        if(!std::strcmp(s,"gesdd"))return 2;
        if(!std::strcmp(s,"gesvd"))return 3;
        throw std::runtime_error("unknown SVD lab mode");}();
    return mode;
}
struct Stats {
    std::atomic<unsigned long long> calls{0},fallbacks{0},diagonal{0},dense{0};
    ~Stats(){std::fprintf(stderr,"SVD_LAB {\"mode\":%d,\"calls\":%llu,\"fallbacks\":%llu,\"diagonal\":%llu,\"dense\":%llu,\"blas_threads\":%d,\"blas_config\":\"%s\"}\n",svd_lab_mode(),calls.load(),fallbacks.load(),diagonal.load(),dense.load(),openblas_get_num_threads(),openblas_get_config());}
};
static Stats stats;
void svd_lab_fallback(){++stats.fallbacks;}
void svd_lab_diagonal(bool x){if(x)++stats.diagonal;else ++stats.dense;}
bool svd_lab_decompose(const float* a,int m,int n,int lda,TwscWorkspace& w) {
    ++stats.calls;
    if(openblas_get_num_threads()!=1)throw std::runtime_error("BLAS must be single threaded");
    int r=std::min(m,n); double maximum=0;
    for(int j=0;j<n;++j)for(int i=0;i<m;++i){double x=a[i+j*lda];if(!std::isfinite(x))return false;maximum=std::max(maximum,std::abs(x));}
    int exponent=0;if(maximum>0)std::frexp(maximum,&exponent);
    double scale=std::ldexp(1.0,-exponent);
    w.qr64.resize(std::size_t(m)*n);
    for(int j=0;j<n;++j)for(int i=0;i<m;++i)w.qr64[i+j*m]=double(a[i+j*lda])*scale;
    // The production QR preserves these column norms (rows for wide matrices).
    double initial=0;
    if(m>=n)for(int j=0;j<n;++j){double z=0;for(int i=0;i<m;++i)z+=w.qr64[i+j*m]*w.qr64[i+j*m];initial=std::max(initial,z);}
    else for(int i=0;i<m;++i){double z=0;for(int j=0;j<n;++j)z+=w.qr64[i+j*m]*w.qr64[i+j*m];initial=std::max(initial,z);}
    w.dictionary.resize(std::size_t(m)*r);w.singular.resize(r);w.vt.resize(std::size_t(r)*n);
    ResourceVector<int> iwork(8*r);
    char job='S';int lwork=-1,info=0;double query=0;
    auto call=[&](double* scratch){
        if(svd_lab_mode()==3)dgesvd_(&job,&job,&m,&n,w.qr64.data(),&m,w.singular.data(),w.dictionary.data(),&m,w.vt.data(),&r,scratch,&lwork,&info);
        else dgesdd_(&job,&m,&n,w.qr64.data(),&m,w.singular.data(),w.dictionary.data(),&m,w.vt.data(),&r,scratch,&lwork,iwork.data(),&info);
    };
    call(&query);
    if(info || !std::isfinite(query) || query<1 || query>100000000)return false;
    lwork=int(query);w.aux64.resize(lwork);call(w.aux64.data());
    if(info)return false;
    for(int k=0;k<r;++k){double s=w.singular[k];w.singular[k]=s*s>initial*1e-14?s/scale:0;
        if(w.singular[k]==0){for(int i=0;i<m;++i)w.dictionary[i+k*m]=0;for(int j=0;j<n;++j)w.vt[k+j*r]=0;}}
    return true;
}
void svd_lab_capture(const float* a,int m,int n) {
    static const char* directory=std::getenv("NSS_SVD_CAPTURE");
    if(!directory)return;
    static std::mutex mutex;
    static std::map<std::pair<int,int>,unsigned long long> counts;
    std::lock_guard<std::mutex> guard(mutex);
    auto k=counts[{m,n}]++;
    // First sixteen, then every 256th group: span successive outer iterations.
    if(k>=16 && k%256!=0)return;
    std::string name=std::string(directory)+"/"+std::to_string(m)+"x"+std::to_string(n)+"-"+std::to_string(k)+".f32";
    FILE* f=std::fopen(name.c_str(),"wb");if(!f)throw std::runtime_error("capture open");
    bool ok=std::fwrite(a,sizeof(float),std::size_t(m)*n,f)==std::size_t(m)*n;
    std::fclose(f);if(!ok)throw std::runtime_error("capture write");
}
}
