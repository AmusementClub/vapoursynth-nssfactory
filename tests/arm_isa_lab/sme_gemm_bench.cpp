// Native Apple SME feasibility only. Includes B packing and per-call mode
// transitions in timing; caller-owned allocation is outside both timed paths.
#include "nss/backend.hpp"
#include "cpu/wnnm/jacobi8.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <vector>

extern "C" unsigned nss_lab_sme_lanes();
extern "C" void nss_lab_sme_gemm(int,int,int,const float*,int,const float*,float*,int);

struct Guard {
    void* map;
    std::size_t bytes;
    float* p;
    int count;
    explicit Guard(int n):count(n) {
        const std::size_t page=sysconf(_SC_PAGESIZE);
        const auto pages=((n+16)*sizeof(float)+page-1)/page;
        bytes=(pages+1)*page;
        map=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
        if(map==MAP_FAILED)std::abort();
        auto end=static_cast<char*>(map)+pages*page;
        if(mprotect(end,page,PROT_NONE))std::abort();
        p=reinterpret_cast<float*>(end)-n;
        std::fill(p-16,p+n,-9876.f);
    }
    ~Guard(){munmap(map,bytes);}
};
int extent(int m,int n,int ld){return n?(n-1)*ld+m:0;}
float value(int x){return float((x*13)%97-48)/128.f;}
void pack(float* out,const float* b,int n,int k,int ldb) {
    // Block the transpose so wide B does not revisit one cache line per column
    // for every scalar row. The complete pack still occurs on every timed call.
    for(int t0=0;t0<k;t0+=32)for(int j0=0;j0<n;j0+=32)
        for(int t=t0;t<std::min(t0+32,k);++t)for(int j=j0;j<std::min(j0+32,n);++j)
            out[t*n+j]=b[j*ldb+t];
}

int main() {
    int supported=0;std::size_t size=sizeof(supported);
    if(sysctlbyname("hw.optional.arm.FEAT_SME",&supported,&size,nullptr,0)||!supported)return 77;
    const auto caps=nss::backend_caps();
    if(!caps.executable||std::strcmp(caps.target_name,"NEON"))return 2;
    const unsigned lanes=nss_lab_sme_lanes();
    int checks=0;
    double worst=0;
    for(int m:{1,3,8,15,16,17,63,64,65,192,256})
        for(int n:{1,3,8,9,16,17,32,33})for(int k:{0,1,7,8,64}) {
            const int lda=m+3,ldb=k+5,ldc=m+7;
            Guard a(extent(m,k,lda)),b(k?extent(k,n,ldb):0),c(extent(m,n,ldc)),bp(k*n);
            for(int t=0;t<k;++t)for(int i=0;i<m;++i)a.p[t*lda+i]=value(i+t);
            for(int j=0;j<n;++j)for(int t=0;t<k;++t)b.p[j*ldb+t]=value(j+t+5);
            pack(bp.p,b.p,n,k,ldb);
            nss_lab_sme_gemm(m,n,k,a.p,lda,bp.p,c.p,ldc);
            for(int j=0;j<n;++j)for(int i=0;i<m;++i) {
                double expected=0;for(int t=0;t<k;++t)expected+=double(a.p[t*lda+i])*b.p[j*ldb+t];
                const double error=std::abs(c.p[j*ldc+i]-expected);worst=std::max(worst,error);
                if(!std::isfinite(c.p[j*ldc+i])||error!=0)return 3;
            }
            for(int i=-16;i<0;++i)if(c.p[i]!=-9876.f)return 4;
            for(int j=0;j+1<n;++j)for(int i=m;i<ldc;++i)if(c.p[j*ldc+i]!=-9876.f)return 5;
            ++checks;
        }
    std::printf("{\"type\":\"correctness\",\"guarded_shapes\":%d,\"max_abs\":%.9g,\"sme_float_lanes\":%u,\"neon_float_lanes\":%d}\n",checks,worst,lanes,caps.float_lanes);
    const int shapes[][3]={{64,1,256},{64,8,8},{192,8,8},{65,9,7},{64,512,256},{256,512,64},
                          {64,947,256},{256,947,64},{64,1444,256},{256,1444,64}};
    for(const auto& shape:shapes) {
        const int m=shape[0],n=shape[1],k=shape[2];
        std::vector<float>a(m*k),b(k*n),bp(k*n),c(m*n),reference(m*n);
        for(int i=0;i<m*k;++i)a[i]=value(i);
        for(int i=0;i<k*n;++i)b[i]=value(i+3);
        nss::gemm_nn_hwy(m,n,k,a.data(),m,b.data(),k,reference.data(),m,false);
        pack(bp.data(),b.data(),n,k,k);
        nss_lab_sme_gemm(m,n,k,a.data(),m,bp.data(),c.data(),m);
        if(std::memcmp(c.data(),reference.data(),c.size()*sizeof(float)))return 6;
        auto run=[&](bool sme,int count) {
            auto start=std::chrono::steady_clock::now();
            for(int r=0;r<count;++r) {
                if(sme){pack(bp.data(),b.data(),n,k,k);nss_lab_sme_gemm(m,n,k,a.data(),m,bp.data(),c.data(),m);}
                else nss::gemm_nn_hwy(m,n,k,a.data(),m,b.data(),k,c.data(),m,false);
                asm volatile(""::"r"(c.data()):"memory");
            }
            return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/count;
        };
        const double calibration=std::max(run(false,10),run(true,10));
        const int repeat=std::max(1,std::min(100000,int(.025/calibration)));
        std::printf("{\"type\":\"pairs\",\"m\":%d,\"n\":%d,\"k\":%d,\"repeat\":%d,\"packing_and_mode_switch_included\":true,\"pairs\":[",m,n,k,repeat);
        for(int p=0;p<15;++p) {
            double neon,sme;
            if(p%2==0){neon=run(false,repeat);sme=run(true,repeat);}
            else{sme=run(true,repeat);neon=run(false,repeat);}
            std::printf("%s[%.12g,%.12g]",p?",":"",neon,sme);
        }
        std::puts("]}");
    }
}
