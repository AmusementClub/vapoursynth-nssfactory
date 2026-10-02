// SPDX-License-Identifier: GPL-2.0-only
// Exact gate comparison against the original scalar arithmetic. Includes
// padded input strides, inactive directions, malformed factors and threshold neighbors.
#include "nss/cpu_twsc_full.hpp"
#include "nss/backend.hpp"
#include "hq_test_target.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
bool reference(const float* a, int m, int n, int lda, const nss::TwscWorkspace& w) {
    const int r=std::min(m,n);double error=0,norm=0,orthogonal=0;
    for(int j=0;j<n;++j) for(int i=0;i<m;++i) {
        double x=0;
        for(int k=0;k<r;++k)x+=w.dictionary[i+k*m]*w.singular[k]*w.vt[k+j*r];
        const double y=a[i+j*lda],delta=x-y;
        if(!std::isfinite(x))return false;
        error+=delta*delta;norm+=y*y;
    }
    for(int j=0;j<r;++j)if(w.singular[j]>0)for(int k=0;k<=j;++k)if(w.singular[k]>0) {
        double dot=0;
        for(int i=0;i<m;++i)dot+=w.dictionary[i+j*m]*w.dictionary[i+k*m];
        orthogonal=std::max(orthogonal,std::abs(dot-(j==k?1.0:0.0)));dot=0;
        for(int i=0;i<n;++i)dot+=w.vt[j+i*r]*w.vt[k+i*r];
        orthogonal=std::max(orthogonal,std::abs(dot-(j==k?1.0:0.0)));
    }
    return error<=norm*2.5e-9 && orthogonal<=5e-4;
}
}
int main(int argc, char** argv) {
    if (!hq_test_target(argc, argv)) return 77;
    try {
        const auto caps = nss::backend_caps();
        for (int columns : {0, 7, 8, 9, 32, 33, 90}) {
            const bool expected = caps.selected_target == HWY_AVX2 ||
                (caps.selected_target == HWY_AVX3 && columns == 8);
            if (nss::twsc_svd_validation_lanes_available(columns) != expected)
                throw std::runtime_error("unexpected TWSC validator target/column capability");
        }
        int checks=0,accepted=0,rejected=0;
        for(auto shape:{std::pair{8,8},{9,8},{49,8},{63,8},{64,8},{192,8},{768,8},
                        {7,8},{64,7},{64,9},{64,32},
                        {32,33},{49,70},{64,90},{192,90},{81,140},{63,33},{768,256},{33,32},{7,70}}) {
            const auto [m,n]=shape;const int lda=m+3,r=std::min(m,n);
            std::vector<float>a(std::size_t(lda)*n,std::numeric_limits<float>::quiet_NaN());
            for(int j=0;j<n;++j)for(int i=0;i<m;++i)a[i+j*lda]=float(std::sin((i+3)*(j+5)*.071)+.01*(i%7));
            nss::TwscWorkspace w;bool fallback=false;
            if(!nss::twsc_svd(a.data(),m,n,lda,w,fallback,true))throw std::runtime_error("fixture SVD failed");
            const auto dictionary=w.dictionary,spectrum=w.singular,vt=w.vt;
            auto check=[&] {
                const bool expected=reference(a.data(),m,n,lda,w);
                if(nss::twsc_valid_svd_lanes(a.data(),m,n,lda,w)!=expected || nss::twsc_valid_svd(a.data(),m,n,lda,w)!=expected)
                    throw std::runtime_error("scalar/lane validation disagreement");
                ++checks;accepted+=expected;rejected+=!expected;
            };
            check();
            for(double delta:{-1e-3,-1e-4,-5e-5,-1e-5,0.0,1e-5,std::nextafter(5e-5,0.0),5e-5,
                              std::nextafter(5e-5,1.0),1e-4,1e-3}) {
                w.dictionary=dictionary;
                for(double& x:w.dictionary)x*=1+delta;
                check();
            }
            for(double bad:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
                w.dictionary=dictionary;w.dictionary[m+r-1]=bad;check();
                w.dictionary=dictionary;w.vt=vt;w.vt[r-1]=bad;check();w.vt=vt;
            }
            w.singular.assign(r,0);w.dictionary=dictionary;std::fill(a.begin(),a.end(),0.f);check();
            w.dictionary[m]=std::numeric_limits<double>::quiet_NaN();check();
            w.dictionary=dictionary;w.vt=vt;w.singular=spectrum;
            a[lda/2]=std::numeric_limits<float>::quiet_NaN();check();
        }
        std::cout<<"{\"passed\":true,\"checks\":"<<checks<<",\"accepted\":"<<accepted
                 <<",\"rejected\":"<<rejected<<",\"target\":\""<<caps.target_name
                 <<"\",\"available_8\":"
                 <<(nss::twsc_svd_validation_lanes_available(8)?"true":"false")<<"}\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
