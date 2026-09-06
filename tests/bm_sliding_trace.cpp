// Replay the exact 32-query host windows around an output discrepancy.
// The first divergence is measured before filtering: direct versus prefix SSD.
#include "nss/cpu_api.hpp"
#include "nss/cpu_batch.hpp"
#include "cpu/bm/sliding-batch.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    const int w=std::atoi(argv[2]), h=std::atoi(argv[3]);
    const int px=std::atoi(argv[4]), py=std::atoi(argv[5]);
    if(w<8 || h<8) return 2;
    std::vector<float> source(static_cast<size_t>(w)*h);
    std::ifstream file(argv[1],std::ios::binary);
    if(!file.read(reinterpret_cast<char*>(source.data()),source.size()*4)) return 3;
    std::vector<nss::MatchBatchItem> jobs;
    for(int y0=0;y0<h-8+2;y0+=2) for(int x0=0;x0<w-8+2;x0+=2)
        jobs.push_back({std::min(x0,w-8),std::min(y0,h-8),8,7,8});
    for(size_t first=0;first<jobs.size();first+=32) {
        const int count=std::min<size_t>(32,jobs.size()-first);
        bool relevant=false;
        for(int i=0;i<count;++i) relevant |= std::abs(jobs[first+i].bx-px)<=15 && std::abs(jobs[first+i].by-py)<=15;
        if(!relevant) continue;
        std::array<nss::Match,32*64> old{},now{};
        std::array<int,32> n0{},n1{};
        if(nss::spatial_match_batch(source.data(),w,w,h,jobs.data()+first,count,old.data(),64,n0.data())) return 4;
        if(nss::detail::sliding_batch(source.data(),w,w,h,jobs.data()+first,count,now.data(),64,n1.data())) return 5;
        for(int i=0;i<count;++i) {
            const auto query=jobs[first+i];
            if(std::abs(query.bx-px)>15 || std::abs(query.by-py)>15) continue;
            std::printf("{\"query\":[%d,%d],\"window_first\":%zu,\"baseline\":[",query.bx,query.by,first);
            for(int k=0;k<n0[i];++k) {
                const auto m=old[i*64+k];
                std::printf(k?",[%d,%d,%.9g]":"[%d,%d,%.9g]",m.x,m.y,m.dist);
            }
            std::printf("],\"candidate\":[");
            for(int k=0;k<n1[i];++k) {
                const auto m=now[i*64+k];
                double exact=0;
                for(int y=0;y<8;++y)for(int x=0;x<8;++x) {
                    const double d=double(source[(query.by+y)*w+query.bx+x])-double(source[(m.y+y)*w+m.x+x]);
                    exact+=d*d;
                }
                const float direct=nss::ssd_block(source.data()+query.by*w+query.bx,w,source.data()+m.y*w+m.x,w,8);
                std::printf(k?",[%d,%d,%.9g,%.9g,%.17g]":"[%d,%d,%.9g,%.9g,%.17g]",m.x,m.y,m.dist,direct,exact);
            }
            std::puts("]}");
        }
    }
}
