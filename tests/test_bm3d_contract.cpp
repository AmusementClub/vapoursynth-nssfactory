#include "nss/cpu_api.hpp"
#include "nss/cpu_mcwnnm.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>
using namespace nss;
void require(bool ok,const char* what){if(!ok) throw std::runtime_error(what);}
void near(double a,double b,double tol,const char* what){if(!(std::abs(a-b)<=tol)){std::fprintf(stderr,"%s %.12g %.12g\n",what,a,b);throw std::runtime_error(what);}}
// Independent double cosine transforms: matrix entries, no production DCT helpers.
void axis(std::vector<double>& v,int length,int stride,int blocks,int span,bool inverse){
 const double pi=std::acos(-1.0);std::vector<double>a(length),b(length);
 for(int base=0;base<blocks;++base)for(int line=0;line<span;++line){
  const int off=base*length*stride+line;
  for(int i=0;i<length;++i)a[i]=v[off+i*stride];
  for(int i=0;i<length;++i){b[i]=0;for(int j=0;j<length;++j){int k=inverse?j:i,x=inverse?i:j;double m=std::sqrt((k?2.:1.)/length)*std::cos(pi*(x+.5)*k/length);b[i]+=m*a[j];}}
  for(int i=0;i<length;++i)v[off+i*stride]=b[i];
 }
}
void transform(std::vector<double>& v,int b,int g,bool inverse){
 if(!inverse){axis(v,b,1,g*b,1,false);axis(v,b,b,g,b,false);axis(v,g,b*b,1,b*b,false);}
 else{axis(v,g,b*b,1,b*b,true);axis(v,b,b,g,b,true);axis(v,b,1,g*b,1,true);}
}
double oracle(std::vector<double>& v,std::vector<double> ref,int b,int g,double sigma,bool wiener){
 transform(v,b,g,false);if(wiener)transform(ref,b,g,false);double weight=0;
 for(std::size_t i=0;i<v.size();++i){if(wiener){double w=i?ref[i]*ref[i]/(ref[i]*ref[i]+sigma*sigma):1.;v[i]*=w;weight+=w*w;}
 else{bool keep=i==0||std::abs(v[i])>=2.7*sigma;if(!keep)v[i]=0;weight+=keep;}}
 transform(v,b,g,true);return 1/std::max(weight,1e-12);
}
void kernels(){
 std::mt19937 rng(42);std::uniform_real_distribution<float> random(-.2f,.8f);
 for(int b:{1,2,4,8,12,16,32})for(int g:{1,2,4,8,16,32,64})for(bool wiener:{false,true}){
  int k=std::min(g,3),area=b*b,lda=area+3;float sigma=.015f;
  std::vector<float>p(g*lda,999),ref(g*lda,999),work(bm3d_filter_work_floats(g,b));
  std::vector<double>v(g*area),r(g*area);
  for(int i=0;i<k;++i)for(int j=0;j<area;++j){v[i*area+j]=p[i*lda+j]=random(rng);r[i*area+j]=ref[i*lda+j]=random(rng);}
  double expected=oracle(v,r,b,g,sigma,wiener);float weight=0;
  bm3d_filter_group(p.data(),lda,g,k,b,sigma,wiener,ref.data(),&weight,work.data());
  near(weight,expected,1e-5,"group weight");
  for(int i=0;i<k;++i){for(int j=0;j<area;++j)near(p[i*lda+j],v[i*area+j],3e-5,"group output");for(int j=area;j<lda;++j)require(p[i*lda+j]==999,"lda padding");}
 }
 for(int k:{1,3,8})for(bool wiener:{false,true}){
  int w=k*8,h=8;std::vector<float>s(w*h),r(w*h),num(w*h),den(w*h);std::vector<double>v(512),rr(512);Match m[8];
  for(int g=0;g<k;++g){m[g]={g*8,0,0,0.f,static_cast<unsigned>(g)};for(int y=0;y<8;++y)for(int x=0;x<8;++x){int i=y*w+g*8+x,j=g*64+y*8+x;v[j]=s[i]=random(rng);rr[j]=r[i]=random(rng);}}
  double weight=oracle(v,rr,8,8,.015f,wiener);
  bm3d_filter8(s.data(),w,m,k,.015f,wiener,r.data(),w,num.data(),den.data(),w,w,h);
  for(int g=0;g<k;++g)for(int y=0;y<8;++y)for(int x=0;x<8;++x){int i=y*w+g*8+x;near(den[i],weight,1e-5,"fused weight");near(num[i]/den[i],v[g*64+y*8+x],3e-5,"fused sigma output");}
 }
}
void matching(){
 float a[64],b[72],c[80];std::fill_n(a,64,1.f);std::fill_n(b,72,0.f);std::fill_n(c,80,2.f);
 const float* refs[]={a,b,c};int strides[]={8,9,10};Match out[64];SearchConfig cfg;cfg.block=8;cfg.group=8;cfg.radius=1;cfg.bm_range=0;cfg.ps_range=0;cfg.ps_num=2;
 int n=predictive_match(refs,strides,3,8,8,0,0,1,cfg,out);require(n==3,"three actual frames");
 for(int i=0;i<n;++i)near(out[i].dist,out[i].t==0?64:out[i].t==1?0:256,0,"cross frame SSD");
 cfg.valid_t_begin=1;n=predictive_match(refs,strides,3,8,8,0,0,1,cfg,out);require(n==2,"clamped slot skipped");cfg.valid_t_begin=0;
 const float* rgb[9]={a,a,a,b,b,b,c,c,c};int st[3]={8,9,10};
 n=predictive_match_nch(rgb,st,3,3,8,8,0,0,1,cfg,out);require(n==3,"RGB unique frames");for(int i=0;i<n;++i)near(out[i].dist,0,0,"static RGB zero distance");
 // Each layer has a distinct translated image; opposite directions diverge.
 int w=40,h=24;std::array<std::vector<float>,5> frames;std::array<const float*,5>p;int ss[]={40,40,40,40,40};
 for(int t=0;t<5;++t){frames[t].resize(w*h);int shift=(t-2)*2;for(int y=0;y<h;++y)for(int x=0;x<w;++x)frames[t][y*w+x]=float(std::sin((x-shift)*.91+y*.73));p[t]=frames[t].data();}
 cfg.block=4;cfg.radius=2;cfg.bm_range=2;cfg.ps_range=2;cfg.ps_num=2;cfg.group=16;
 n=predictive_match(p.data(),ss,5,w,h,16,8,2,cfg,out);std::set<std::tuple<int,int,int>> unique;bool found[5]{};
 for(int i=0;i<n;++i){auto m=out[i];require(unique.emplace(m.t,m.y,m.x).second,"duplicate window candidate");double d=0;for(int y=0;y<4;++y)for(int x=0;x<4;++x){double e=frames[2][(8+y)*w+16+x]-frames[m.t][(m.y+y)*w+m.x+x];d+=e*e;}near(m.dist,d,1e-4,"motion SSD oracle");if(m.y==8&&m.x==16+(m.t-2)*2)found[m.t]=true;}
 for(bool f:found)require(f,"directional motion advance");
}
void aggregation(){
 for(int count:{1,2,3,9,33}){int w=19,h=3,ds=23,ss=25;std::vector<std::vector<float>> n(count),d(count);std::vector<const float*>np(count),dp(count);std::vector<int>st(count);std::vector<float>src(ss*h,.27f),dst(ds*h,-9.f);
  for(int c=0;c<count;++c){st[c]=w+c+1;n[c].resize(st[c]*h);d[c].resize(st[c]*h);for(int y=0;y<h;++y)for(int x=0;x<w;++x){n[c][y*st[c]+x]=float((c+1)*(x+y));d[c][y*st[c]+x]=x?float(c+1):0;}np[c]=n[c].data();dp[c]=d[c].data();}
  vaggregate_target(dst.data(),np.data(),dp.data(),st.data(),count,src.data(),w,h,ds,ss);
  for(int y=0;y<h;++y){for(int x=0;x<w;++x)near(dst[y*ds+x],x?x+y:.27f,1e-6,"target slices/stride/fallback");for(int x=w;x<ds;++x)require(dst[y*ds+x]==-9.f,"output padding");}
 }
}
int main(){try{matching();aggregation();kernels();std::puts("BM3D independent contracts passed");return 0;}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
