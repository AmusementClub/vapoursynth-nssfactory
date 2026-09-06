// Real spatial distance streams, including initial fill, rejection and cutoff ties.
// Diagnostic only; whole-filter paired measurements determine performance.
#include "nss/cpu_api.hpp"
#include "cpu/bm/matcher.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const int w=std::atoi(argv[2]), h=std::atoi(argv[3]);
    if (w<8 || h<8) return 2;
    std::vector<float> src(static_cast<size_t>(w)*h);
    std::ifstream file(argv[1],std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(src.data()),src.size()*sizeof(float))) return 3;
    for (int group : {2,4,8,16,32,64}) {
        std::uint64_t accepted=0,rejected=0,ties=0,fill=0,windows=0;
        for (int y=0; y<h-7; y+=8) for (int x=0; x<w-7; x+=8) {
            nss::Match stable[64], sorted[64], cached[64];
            nss::detail::StableTopK a(stable,group-1);
            nss::detail::SpatialSortedTopK b(sorted,group-1);
            nss::detail::CachedWorstTopK c(cached,group-1);
            std::vector<nss::Match> tracked;
            std::uint32_t ordinal=1;
            for(int yy=std::max(0,y-7); yy<=std::min(h-8,y+7); ++yy)
            for(int xx=std::max(0,x-7); xx<=std::min(w-8,x+7); ++xx,++ordinal) {
                if(xx==x && yy==y) continue;
                const float d=nss::ssd_block(src.data()+y*w+x,w,src.data()+yy*w+xx,w,8);
                const nss::Match m{xx,yy,0,d,ordinal};
                a.add(m);b.add(m);c.add(m);
                if(tracked.size()<static_cast<size_t>(group-1)){tracked.push_back(m);++fill;}
                else {
                    const auto worst=std::max_element(tracked.begin(),tracked.end(),nss::detail::match_less);
                    if(d==worst->dist) ++ties;
                    if(nss::detail::match_less(m,*worst)){*worst=m;++accepted;}else ++rejected;
                }
            }
            const int n=a.finish();
            if(b.finish()!=n || c.finish()!=n) return 4;
            for(int i=0;i<n;++i) for(auto other:{sorted[i],cached[i]})
                if(stable[i].x!=other.x || stable[i].y!=other.y || stable[i].dist!=other.dist || stable[i].ordinal!=other.ordinal) return 5;
            ++windows;
        }
        std::cout<<"{\"group\":"<<group<<",\"windows\":"<<windows<<",\"initial_fill\":"<<fill
                 <<",\"accepted\":"<<accepted<<",\"rejected\":"<<rejected<<",\"cutoff_ties\":"<<ties
                 <<",\"identical_sorted_results\":true}\n";
    }
}
