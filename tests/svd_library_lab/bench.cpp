// SPDX-License-Identifier: GPL-2.0-only
#include "nss/cpu_twsc_full.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>
using Clock=std::chrono::steady_clock;
template<class F> double timed(F f,int loops){auto t=Clock::now();for(int i=0;i<loops;++i)f();return std::chrono::duration<double>(Clock::now()-t).count()/loops;}
int main(int argc,char** argv){try{
    if(argc!=8)throw std::runtime_error("m n count loops sigma input output-prefix");
    int m=std::stoi(argv[1]),n=std::stoi(argv[2]),count=std::stoi(argv[3]),loops=std::stoi(argv[4]),r=std::min(m,n);
    double sigma=std::stod(argv[5])/255.;
    if(m<1||m>768||n<1||n>256||count<1||count>16||loops<1)throw std::runtime_error("shape");
    std::vector<float> a(std::size_t(m)*n*count),out(a.size());
    std::ifstream in(argv[6],std::ios::binary);in.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(float));
    if(!in || in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("input size");
    std::vector<nss::TwscWorkspace> ws(count);std::vector<nss::TwscWorkspace*> ptr(count);
    std::vector<nss::ResourceVector<double>> saved(count);
    for(int k=0;k<count;++k){auto& w=ws[k];w.input.assign(a.begin()+std::size_t(k)*m*n,a.begin()+std::size_t(k+1)*m*n);w.centered.assign(w.input.begin(),w.input.end());w.mean.assign(m,0);ptr[k]=&w;}
    nss::ResourceVector<float> rs(m,float(sigma)),cs(n,float(sigma)),weights(n);
    auto svd=[&]{nss::twsc_svd64_batch(ptr.data(),m,n,count);};
    auto valid=[&]{for(auto& w:ws)if(!nss::twsc_valid_svd(w.input.data(),m,n,m,w))throw std::runtime_error("validation");};
    auto save=[&]{for(int k=0;k<count;++k)saved[k]=ws[k].singular;};
    auto finish=[&]{for(int k=0;k<count;++k){ws[k].singular=saved[k];nss::twsc_finish_full(out.data()+std::size_t(k)*m*n,m,n,m,rs.data(),cs.data(),weights.data(),{},ws[k],true);}};
    svd();valid();save();finish();svd();
    double decomposition=timed(svd,loops);valid();
    double validation=timed(valid,loops);save();finish();
    double coding=timed(finish,loops);
    double total=timed([&]{svd();valid();save();finish();},loops);
    svd();valid();
    std::ofstream factors(std::string(argv[7])+".factors",std::ios::binary);
    for(auto& w:ws)for(auto* v:{&w.dictionary,&w.singular,&w.vt})factors.write(reinterpret_cast<const char*>(v->data()),v->size()*sizeof(double));
    save();finish();std::ofstream output(std::string(argv[7])+".f32",std::ios::binary);output.write(reinterpret_cast<const char*>(out.data()),out.size()*sizeof(float));
    std::cout<<std::setprecision(17)<<"{\"m\":"<<m<<",\"n\":"<<n<<",\"count\":"<<count<<",\"loops\":"<<loops<<",\"decomposition_s\":"<<decomposition<<",\"validation_s\":"<<validation<<",\"coding_s\":"<<coding<<",\"total_s\":"<<total<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
