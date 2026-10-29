// SPDX-License-Identifier: GPL-2.0-only
#include "nss/cpu_common.hpp"
#include "cpu/hwy_config.hpp"
#include "hq_test_target.hpp"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "test_nlh_top3.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
HWY_BEFORE_NAMESPACE();
namespace nss { namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
#include "cpu/nlh/select_top3-inl.hpp"
int Top3Check() {
    const hn::ScalableTag<float> d;
    std::mt19937 rng(20260929);
    int accepted=0, fallback=0;
    for(int kind=0;kind<16;++kind) for(int seed=0;seed<16;++seed) for(int self=0;self<64;++self) {
        std::array<float,64> row;
        for(int j=0;j<64;++j) {
            float value=float(rng()%1024)/128.f;
            if(kind==0) value=0;
            if(kind==1) value=float(j%3);
            if(kind==2) value=float((j+seed)%64);
            if(kind==3) value=std::nextafter(1e30f,0.f);
            if(kind==4) value=(j%2)?0.f:std::numeric_limits<float>::denorm_min();
            row[j]=value;
        }
        row[self]=0;
        const int special=(self+seed+1)%64;
        if(kind==5) row[special]=1e30f;
        if(kind==6) row[special]=std::nextafter(1e30f,std::numeric_limits<float>::infinity());
        if(kind==7) row[special]=std::numeric_limits<float>::infinity();
        if(kind==8) row[special]=std::numeric_limits<float>::quiet_NaN();
        if(kind==9) row[special]=-0.f;
        if(kind==10) row[special]=-1.f;
        if(kind==11) { row.fill(1e30f);row[self]=0;row[(self+1)%64]=0; }
        const auto original=row;
        std::array<int,5> output{-321,-321,-321,-321,-321};
        std::array<float,5> distances{-321,-321,-321,-321,-321};
        const bool ok=SelectTopThree64(d,row.data(),self,output.data()+1,distances.data()+1);
        bool eligible=true;
        for(float x:row) eligible &= std::bit_cast<std::uint32_t>(x)<std::bit_cast<std::uint32_t>(1e30f);
        if(ok!=eligible) throw std::runtime_error("incorrect exceptional-row eligibility");
        if(output.front()!=-321 || output.back()!=-321 || distances.front()!=-321 || distances.back()!=-321)
            throw std::runtime_error("output guard overwritten");
        for(int i=0;i<64;++i) if(std::bit_cast<std::uint32_t>(row[i])!=std::bit_cast<std::uint32_t>(original[i]))
            throw std::runtime_error("distance row modified");
        if(!ok) {
            ++fallback;
            for(int i=1;i<=3;++i) if(output[i]!=-321 || distances[i]!=-321) throw std::runtime_error("fallback committed output");
            continue;
        }
        ++accepted;
        std::array<std::pair<float,int>,63> sorted;
        int k=0;for(int i=0;i<64;++i) if(i!=self) sorted[k++]={row[i],i};
        std::sort(sorted.begin(),sorted.end());
        for(int i=0;i<3;++i) if(output[i+1]!=sorted[i].second || distances[i+1]!=sorted[i].first)
            throw std::runtime_error("top-three distance/index ordering changed");
    }
    std::cout<<"rows accepted="<<accepted<<" fallback="<<fallback<<"\n";
    return 0;
}
}}
HWY_AFTER_NAMESPACE();
#if HWY_ONCE
namespace nss { HWY_EXPORT(Top3Check); int run_top3_check() { return HWY_DYNAMIC_DISPATCH(Top3Check)(); } }
int main(int argc,char** argv) {
    if(!hq_test_target(argc,argv)) return 77;
    try {return nss::run_top3_check();}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
#endif
