#include "nss/cpu_batch.hpp"
#include "hq_test_target.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

int main(int argc,char** argv) {
    if(!hq_test_target(argc,argv)) return 77;
    std::mt19937 rng(512);
    for(int block:{1,4,6,7,8,9,12,16})for(int extra:{0,1,9,23})for(int group:{1,2,8,16,32,64})
    for(int kind=0;kind<5;++kind)for(int range:{0,1,7,30}) {
        const int width=block+extra,height=block+extra/2,stride=width+7;
        std::vector<float> image(height*stride,std::numeric_limits<float>::quiet_NaN());
        for(int y=0;y<height;++y)for(int x=0;x<width;++x)
            image[y*stride+x]=kind==0?.5f:kind==1?float((x+y)%3)/8:
                kind==4?.5f+.24f*std::sin((y*width+x)*.71f)+.07f*std::cos(x*.123f):float(int(rng()%1024)-512)/1024;
        if(kind==3) image[0]=std::numeric_limits<float>::quiet_NaN();
        std::array<nss::MatchBatchItem,7> items{};
        for(int i=0;i<7;++i)items[i]={i*extra/6,i*(height-block)/6,block,range,group,0};
        std::array<nss::Match,7*64> got{},want{};std::array<int,7> counts{};
        if(nss::spatial_match_joint(image.data(),stride,width,height,items.data(),7,got.data(),64,counts.data())) return 1;
        for(int i=0;i<7;++i) {
            const int n=nss::spatial_match(image.data(),stride,width,height,items[i].bx,items[i].by,block,range,group,want.data()+i*64,0);
            if(n!=counts[i]) return 2;
            for(int j=0;j<n;++j) {
                const auto& a=got[i*64+j];const auto& b=want[i*64+j];
                if(a.x!=b.x || a.y!=b.y || a.t!=b.t || a.ordinal!=b.ordinal ||
                   (std::memcmp(&a.dist,&b.dist,sizeof(float)) && !(std::isnan(a.dist)&&std::isnan(b.dist)))) {
                    std::cerr<<"b="<<block<<" g="<<group<<" extra="<<extra<<" kind="<<kind<<" range="<<range
                             <<" job="<<i<<" rank="<<j<<" dist="<<a.dist<<" expected="<<b.dist<<"\n";return 3;
                }
            }
        }
    }
    std::cout<<"joint matches: exact same-ISA distances, candidate order, tails and ties PASS\n";
}
