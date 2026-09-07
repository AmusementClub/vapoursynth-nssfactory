#ifdef NSS_TEMPORAL_TEST_MASK
#undef NSS_BM_EXPERIMENT
#define NSS_BM_EXPERIMENT NSS_TEMPORAL_TEST_MASK
#include "cpu/bm/matcher.hpp"
#endif
#include "nss/cpu_api.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
// TEST ONLY: independent finite-data predictive-search oracle; no production comparator.
#include <set>
#include <tuple>
#include <stdexcept>
bool less_ref(const nss::Match&a,const nss::Match&b){
 bool af=std::isfinite(a.dist),bf=std::isfinite(b.dist);if(af!=bf)return af;
 if(af&&a.dist!=b.dist)return a.dist<b.dist;
 return std::tuple(a.ordinal!=0,a.t,a.y,a.x,a.ordinal)<std::tuple(b.ordinal!=0,b.t,b.y,b.x,b.ordinal);
}
std::vector<nss::Match> reference(const std::vector<const float*>& refs,const std::vector<int>& strides,
 int nc,int nt,int w,int h,int bx,int by,int t0,const nss::SearchConfig& cfg,
 std::set<std::tuple<int,int,int>>* visited=nullptr){
 int b=cfg.block,cx=std::clamp(bx,0,w-b),cy=std::clamp(by,0,h-b);
 auto distance=[&](int t,int x,int y){if(visited)visited->emplace(t,y,x);float v=0;for(int c=0;c<nc;++c){float ch=0;for(int j=0;j<b;++j)for(int i=0;i<b;++i){float e=refs[c*nt+t0][(cy+j)*strides[c]+cx+i]-refs[c*nt+t][(y+j)*strides[c]+x+i];ch+=e*e;}v+=ch;}return v;};
 std::vector<nss::Match> spatial{{cx,cy,t0,0,0}};
 if(cfg.group==1)return spatial;
 for(int y=std::max(0,cy-cfg.bm_range);y<=std::min(h-b,cy+cfg.bm_range);++y)
 for(int x=std::max(0,cx-cfg.bm_range);x<=std::min(w-b,cx+cfg.bm_range);++x)
 if(x!=cx||y!=cy)spatial.push_back({x,y,t0,distance(t0,x,y),1});
 std::sort(spatial.begin(),spatial.end(),less_ref);if(spatial.size()>unsigned(cfg.group))spatial.resize(cfg.group);
 if(cfg.radius==0||nt==1||cfg.group==1)return spatial;
 std::vector<nss::Match> all=spatial;
 for(int sign:{-1,1}){
  std::vector<nss::Match> seeds(spatial.begin(),spatial.begin()+std::min<int>(spatial.size(),cfg.ps_num));
  for(int delta=1;delta<=cfg.radius;++delta){int t=t0+sign*delta;
   if(t<cfg.valid_t_begin||t>=(cfg.valid_t_end<0?nt:cfg.valid_t_end))break;
   std::set<std::pair<int,int>> coords;
   for(auto p:seeds)for(int y=std::max(0,p.y-cfg.ps_range);y<=std::min(h-b,p.y+cfg.ps_range);++y)
     for(int x=std::max(0,p.x-cfg.ps_range);x<=std::min(w-b,p.x+cfg.ps_range);++x)coords.emplace(y,x);
   std::vector<nss::Match> layer;for(auto [y,x]:coords)layer.push_back({x,y,t,distance(t,x,y),1});
   std::sort(layer.begin(),layer.end(),less_ref);if(layer.size()>unsigned(std::min(cfg.ps_num,cfg.group)))layer.resize(std::min(cfg.ps_num,cfg.group));
   all.insert(all.end(),layer.begin(),layer.end());seeds=layer;if(seeds.empty())break;
  }
 }
 std::sort(all.begin(),all.end(),less_ref);if(all.size()>unsigned(cfg.group))all.resize(cfg.group);return all;
}
#ifdef NSS_TEMPORAL_TEST_MASK
int production_control(const std::vector<const float*>& refs, const std::vector<int>& strides,
                       int nc, int nt, int w, int h, int bx, int by, int t0, const nss::SearchConfig& cfg,
                       nss::Match* out, std::set<std::tuple<int,int,int>>& visited) {
    const int cx=std::clamp(bx,0,w-cfg.block), cy=std::clamp(by,0,h-cfg.block);
    auto distance=[&](int t,int x,int y) {
        visited.emplace(t,y,x);
        const float* query[3]; const float* candidate[3];
        for(int c=0;c<nc;++c) {
            query[c]=refs[c*nt+t0]+cy*strides[c]+cx;
            candidate[c]=refs[c*nt+t]+y*strides[c]+x;
        }
        return nc==1 ? nss::ssd_block(query[0],strides[0],candidate[0],strides[0],cfg.block)
                     : nss::ssd_nch(query,strides.data(),candidate,strides.data(),nc,cfg.block);
    };
    int count=nss::detail::collect_spatial_coords(w,h,bx,by,cfg.block,cfg.bm_range,cfg.group,out,
                                                 [&](int,int,int x,int y){return distance(t0,x,y);});
    return nss::detail::collect_temporal(w,h,t0,nt,cfg,out,count,distance);
}
#endif
int main(){std::mt19937 rng(3917);long candidates=0,visits=0;
 for(int test=0;test<1500;++test){
  int nc=test%3+1,b=std::array<int,6>{1,2,4,8,12,16}[rng()%6],w=b+int(rng()%12),h=b+int(rng()%9),r=rng()%4,nt=2*r+1,t0=r;
  nss::SearchConfig cfg;cfg.block=b;cfg.group=1<<(rng()%7);cfg.radius=r;cfg.bm_range=rng()%5;cfg.ps_num=1+rng()%std::min(8,cfg.group);cfg.ps_range=rng()%4;
  cfg.valid_t_begin=rng()%(r+1);cfg.valid_t_end=t0+1+rng()%(r+1);
  std::vector<std::vector<float>> mem(nc*nt);std::vector<const float*> refs(nc*nt);std::vector<int> strides(nc);
  for(int c=0;c<nc;++c){strides[c]=w+2+c;for(int t=0;t<nt;++t){auto&v=mem[c*nt+t];v.resize(strides[c]*h);for(auto&f:v)f=float(int(rng()%9)-4)*.125f;refs[c*nt+t]=v.data();}}
  int bx=int(rng()%(w+4))-2,by=int(rng()%(h+4))-2;nss::Match out[64];int ns;
  std::set<std::tuple<int,int,int>> actual_visits, expected_visits;
#ifdef NSS_TEMPORAL_TEST_MASK
  ns=production_control(refs,strides,nc,nt,w,h,bx,by,t0,cfg,out,actual_visits);
#else
  if(nc==1){std::vector<int> st(nt,strides[0]);ns=nss::predictive_match(refs.data(),st.data(),nt,w,h,bx,by,t0,cfg,out);}
  else ns=nss::predictive_match_nch(refs.data(),strides.data(),nc,nt,w,h,bx,by,t0,cfg,out);
#endif
  auto want=reference(refs,strides,nc,nt,w,h,bx,by,t0,cfg,&expected_visits);
#ifdef NSS_TEMPORAL_TEST_MASK
  if(actual_visits!=expected_visits){std::fprintf(stderr,"visit test=%d nc=%d b=%d g=%d r=%d actual=%zu expected=%zu\n",test,nc,b,cfg.group,r,actual_visits.size(),expected_visits.size());return 3;}
#endif
  visits+=expected_visits.size();if(ns!=int(want.size()))throw std::runtime_error("count mismatch");
  std::set<std::tuple<int,int,int>> unique;
  for(int i=0;i<ns;++i){const auto&a=out[i];const auto&z=want[i];if(!unique.emplace(a.t,a.y,a.x).second||a.t!=z.t||a.x!=z.x||a.y!=z.y||a.dist!=z.dist){std::fprintf(stderr,"test=%d i=%d nc=%d b=%d g=%d radius=%d got=(%d,%d,%d,%g) want=(%d,%d,%d,%g)\n",test,i,nc,b,cfg.group,r,a.t,a.y,a.x,a.dist,z.t,z.y,z.x,z.dist);return 2;}++candidates;}
 }
 std::printf("predictive independent union-window oracle cases=1500 retained=%ld visited=%ld mismatches=0\n",candidates,visits);
}
