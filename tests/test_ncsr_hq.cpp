#include "nss/ncsr_hq.hpp"
#include "nss/ncsr_alignment_lab.hpp"
#include "nss/resources.hpp"
#include "hq_test_target.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <vector>

static void require(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
static std::vector<float> check(int block,int group,int iterations,float sigma,bool flat=false) {
    const int w=29,h=27,stride=w+3;
    std::vector<float> src(h*stride,-17),ref(h*stride,-19),out=ref;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)
        src[y*stride+x]=flat?.5f:.5f+.24f*std::sin((y*w+x)*.71f)+.07f*std::cos(x*.123f);
    nss::SearchConfig cfg;cfg.block=block;cfg.step=std::min(3,block);cfg.group=group;cfg.bm_range=4;cfg.radius=0;
    std::string a,b;
    nss::alignment_lab::denoise(src.data(),w,h,stride,ref.data(),stride,cfg,sigma/255,iterations,.02f,31,a);
    nss::ncsr_hq::denoise(src.data(),w,h,stride,out.data(),stride,cfg,sigma/255,iterations,.02f,b);
    if(std::memcmp(ref.data(),out.data(),out.size()*sizeof(float))) {
        float max_error=0;int changed=0;
        for(std::size_t i=0;i<ref.size();++i) {max_error=std::max(max_error,std::abs(ref[i]-out[i]));changed+=ref[i]!=out[i];}
        std::cerr<<"block="<<block<<" group="<<group<<" iters="<<iterations<<" sigma="<<sigma
                 <<" changed="<<changed<<" max_error="<<max_error<<"\n";
        throw std::runtime_error("HQ scalar reference differs");
    }
    return out;
}
int main(int argc,char** argv) {
    if(!hq_test_target(argc,argv)) return 77;
    try {
        for(int b:{6,7,8,9})for(int g:{1,8,16,32}) check(b,g,4,25);
        for(float sigma:{0.f,5.f,15.f,50.f}) check(7,16,9,sigma);
        check(7,32,4,25,true);
        for(int b:{1,4,12,16}) check(b,8,4,25);
        auto first=std::async(std::launch::async,[]{return check(7,16,4,15);});
        auto second=std::async(std::launch::async,[]{return check(9,8,4,50);});
        require(first.get()==check(7,16,4,15),"concurrent first differs");
        require(second.get()==check(9,8,4,50),"concurrent second differs");
        auto budget=std::make_shared<nss::ResourceBudget>(256);
        bool failed=false;
        { nss::ResourceScope scope(budget);
          try {std::vector<float> in(64,.5f),out(64);std::string trace;nss::SearchConfig cfg;cfg.block=7;cfg.step=1;
              nss::ncsr_hq::denoise(in.data(),8,8,8,out.data(),8,cfg,.1f,1,.02f,trace);
          } catch(const std::exception&) {failed=true;}
        }
        require(failed,"resource limit not enforced");require(budget->snapshot().owned==0,"failure leaked budget");
        for(std::size_t limit:{1024u,16384u,65536u,262144u,1048576u,16777216u}) {
            auto bounded=std::make_shared<nss::ResourceBudget>(limit);
            { nss::ResourceScope scope(bounded);
              try { check(7,16,4,25); } catch(const std::runtime_error& error) {
                  require(std::string(error.what()).find("memory_limit_mb")!=std::string::npos,"unexpected budget failure");
              }
            }
            require(bounded->snapshot().owned==0,"mid-pipeline failure leaked budget");
            require(bounded->snapshot().peak<=limit,"budget peak exceeded limit");
        }
        { std::vector<float> input(64,.5f),output(64,-3);std::string trace;nss::SearchConfig cfg;cfg.block=7;cfg.step=1;
          input[0]=std::numeric_limits<float>::quiet_NaN();bool rejected=false;
          try {nss::ncsr_hq::denoise(input.data(),8,8,8,output.data(),8,cfg,.1f,1,.02f,trace);}
          catch(const std::invalid_argument&) {rejected=true;}
          require(rejected,"nonfinite input accepted");
          require(std::all_of(output.begin(),output.end(),[](float v){return v==-3;}),"failed call wrote output");
        }
        std::cout<<"HQ exact oracle, padded strides, flat ties, generations, concurrency and budget PASS\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
