#include "nss/avx2_policy.hpp"
// Fixed-match replay around a failing pixel; serializable inputs, outputs and weights.
#include "nss/cpu_api.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
void floats(const float*p,int n){std::putchar('[');for(int i=0;i<n;++i)std::printf(i?",%.9g":"%.9g",p[i]);std::putchar(']');}
int main(int argc,char**v){if(argc!=10)return 2;int w=std::atoi(v[2]),h=std::atoi(v[3]),b=std::atoi(v[4]),g=std::atoi(v[5]),step=std::atoi(v[6]),px=std::atoi(v[8]),py=std::atoi(v[9]);float sigma=std::atof(v[7]);const unsigned features=nss::detail::avx2_policy(nss::detail::Avx2Algorithm::BM3D,b,g,0);int area=b*b;std::vector<float>src(w*h);std::ifstream f(v[1],std::ios::binary);if(!f.read(reinterpret_cast<char*>(src.data()),src.size()*4))return 3;
 for(int by0=0;by0<h-b+step;by0+=step)for(int bx0=0;bx0<w-b+step;bx0+=step){int x=std::min(bx0,w-b),y=std::min(by0,h-b);if(std::abs(x-px)>b+7||std::abs(y-py)>b+7)continue;nss::Match m[64];int k=nss::spatial_match(src.data(),w,w,h,x,y,b,7,g,m,features);bool covers=false;for(int j=0;j<k;++j)covers|=m[j].x<=px&&px<m[j].x+b&&m[j].y<=py&&py<m[j].y+b;if(!covers)continue;
 std::vector<float>p(g*area),work(nss::bm3d_filter_work_floats(g,b));for(int j=0;j<k;++j)nss::pack_patch(p.data()+j*area,area,src.data(),w,m[j].x,m[j].y,b,w,h);auto input=p;float weight=0;nss::bm3d_filter_group(p.data(),area,g,k,b,sigma,false,nullptr,&weight,work.data(),features);
 std::printf("{\"query\":[%d,%d],\"k\":%d,\"group\":%d,\"block\":%d,\"sigma_eff\":%.9g,\"weight\":%.9g,\"matches\":[",x,y,k,g,b,sigma,weight);for(int j=0;j<k;++j)std::printf(j?",[%d,%d,%.9g]":"[%d,%d,%.9g]",m[j].x,m[j].y,m[j].dist);std::printf("],\"input\":");floats(input.data(),g*area);std::printf(",\"output\":");floats(p.data(),g*area);std::puts("}");}
}
