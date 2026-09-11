// Linux-only, explicitly preloaded diagnostic. Counts shapes at the frame
// boundary; never use its instrumented elapsed time for performance claims.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <mutex>
#include <tuple>

namespace {
std::atomic<bool> enabled{false};
std::mutex lock;
std::map<std::tuple<int,int,int,int>,unsigned long long> counts;
void* resolve(const char* symbol) {
    const char* path=std::getenv("NSS_GEMM_TRACE_PLUGIN");
    void* handle=path?dlopen(path,RTLD_NOLOAD|RTLD_NOW):nullptr;
    void* result=handle?dlsym(handle,symbol):nullptr;
    if(!result){std::fprintf(stderr,"GEMM trace cannot resolve %s: %s\n",symbol,dlerror());std::abort();}
    return result;
}
void count(int kind,int m,int n,int k) {
    if(!enabled.load(std::memory_order_relaxed))return;
    std::lock_guard<std::mutex> guard(lock);
    ++counts[{kind,m,n,k}];
}
}
extern "C" void nss_gemm_trace_enable(int value) { enabled.store(value!=0); }
extern "C" void nss_gemm_trace_save(const char* path) {
    std::lock_guard<std::mutex> guard(lock);
    FILE* f=std::fopen(path,"w");if(!f)std::abort();
    std::fputs("kind,m,n,k,calls,nominal_fp_ops\n",f);
    for(const auto& [shape,calls]:counts) {
        const auto [kind,m,n,k]=shape;
        std::fprintf(f,"%s,%d,%d,%d,%llu,%llu\n",kind?"TN":"NN",m,n,k,calls,
                     2ull*m*n*k*calls);
    }
    std::fclose(f);
}
namespace nss {
void gemm_nn_hwy(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc,bool avx2) {
    using F=void(*)(int,int,int,const float*,int,const float*,int,float*,int,bool);
    static auto real=reinterpret_cast<F>(resolve("_ZN3nss11gemm_nn_hwyEiiiPKfiS1_iPfib"));
    count(0,m,n,k);real(m,n,k,a,lda,b,ldb,c,ldc,avx2);
}
void gemm_tn_hwy(int m,int n,int k,const float* a,int lda,const float* b,int ldb,float* c,int ldc) {
    using F=void(*)(int,int,int,const float*,int,const float*,int,float*,int);
    static auto real=reinterpret_cast<F>(resolve("_ZN3nss11gemm_tn_hwyEiiiPKfiS1_iPfi"));
    count(1,m,n,k);real(m,n,k,a,lda,b,ldb,c,ldc);
}
}
