#include "nss/cpu_linalg.hpp"
#include "hq_test_target.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

int main(int argc,char** argv) {
    if(!hq_test_target(argc,argv)) return 77;
    std::mt19937 rng(329);
    for(int m:{1,7,36,49,81})for(int n:{1,2,3,7,16,33,127})for(int k:{0,1,7,49,81}) {
        const int lda=k+3,ldb=n+5,ldc=n+7;
        std::vector<float> a(m*lda),b(k*ldb),c(m*ldc,-19),ref=c;
        for(auto& v:a)v=std::ldexp(float(int(rng()%2048)-1024),int(rng()%16)-12);
        for(auto& v:b)v=std::ldexp(float(int(rng()%2048)-1024),int(rng()%16)-12);
        for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
            double sum=0;for(int t=0;t<k;++t) {const float product=a[i*lda+t]*b[t*ldb+j];sum+=product;}
            ref[i*ldc+j]=static_cast<float>(sum);
        }
        if(nss::gemm_f32_f64(a.data(),lda,b.data(),ldb,c.data(),ldc,m,n,k))return 1;
        if(std::memcmp(ref.data(),c.data(),c.size()*sizeof(float))) {
            std::cerr<<"GEMM mismatch m="<<m<<" n="<<n<<" k="<<k<<"\n";return 2;
        }
    }
    if(nss::gemm_f32_f64(nullptr,0,nullptr,0,nullptr,0,0,0,0))return 3;
    if(nss::gemm_f32_f64(nullptr,1,nullptr,1,nullptr,1,1,1,1)!=-1)return 4;
    std::cout<<"float products, ordered double accumulation, padded views and tails exact PASS\n";
}
