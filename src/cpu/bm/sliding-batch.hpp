#pragma once
#include "cpu/bm/matcher.hpp"
#include "nss/cpu_batch.hpp"
#include <limits>
#include <vector>
namespace nss::detail {
inline int sliding_batch(const float* image,int stride,int w,int h,const MatchBatchItem* items,int count,Match* out,int os,int* counts){
    if(count<1||count>32)return -2;
    int b=items[0].block,range=items[0].bm_range,group=items[0].group;
    if(b<1||w<b||h<b||group<1||group>64||os<group)return -2;
    int x0=w,y0=h,x1=0,y1=0;
    for(int q=0;q<count;++q){auto a=items[q];if(a.block!=b||a.bm_range!=range||a.group!=group||a.bx<0||a.by<0||a.bx>w-b||a.by>h-b)return -2;x0=std::min(x0,a.bx);y0=std::min(y0,a.by);x1=std::max(x1,a.bx+b);y1=std::max(y1,a.by+b);}
    for(int y=std::max(0,y0-range);y<std::min(h,y1+range);++y)
        for(int x=std::max(0,x0-range);x<std::min(w,x1+range);++x)
            if(!finite_distance(image[y*stride+x]))return -2;
    int tw=x1-x0,th=y1-y0,ps=tw+1;
    if(static_cast<std::size_t>(ps)*(th+1)>512*1024)return -2;
    std::vector<double> prefix(static_cast<std::size_t>(ps)*(th+1));
    std::vector<CachedWorstTopK> top;
    top.reserve(count);
    for(int q=0;q<count;++q){out[q*os]={items[q].bx,items[q].by,0,0.f,0};top.emplace_back(out+q*os+1,group-1);}
    for(int dy=-range;dy<=range;++dy)for(int dx=-range;dx<=range;++dx){
        if(dx==0&&dy==0)continue;
        std::fill_n(prefix.data(),ps,0.);
        for(int y=0;y<th;++y){double row=0;prefix[(y+1)*ps]=0;
            for(int x=0;x<tw;++x){int ax=x+x0,ay=y+y0,bx=ax+dx,by=ay+dy;double d=0;if(bx>=0&&bx<w&&by>=0&&by<h){d=double(image[ay*stride+ax])-image[by*stride+bx];}row+=d*d;prefix[(y+1)*ps+x+1]=prefix[y*ps+x+1]+row;}}
        for(int q=0;q<count;++q){int cx=items[q].bx+dx,cy=items[q].by+dy;if(cx<0||cy<0||cx>w-b||cy>h-b)continue;int x=items[q].bx-x0,y=items[q].by-y0;
            double d=prefix[(y+b)*ps+x+b]-prefix[y*ps+x+b]-prefix[(y+b)*ps+x]+prefix[y*ps+x];
            // Double accumulation and a fresh prefix per displacement avoid drift.
            top[q].add(Match{cx,cy,0,static_cast<float>(std::max(0.,d)),1});
        }
    }
    for(int q=0;q<count;++q)counts[q]=1+top[q].finish();return 0;
}
}
