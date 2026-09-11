#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"
#include "cpu/lssc/mixed_omp.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
extern nss::detail::MixedOmpStats mixed_stats;
int main(int argc,char**argv) {
    if(argc!=6) return 2;
    const int w=std::atoi(argv[2]),h=std::atoi(argv[3]),b=std::atoi(argv[4]);
    std::vector<float> source(w*h),num(w*h),den(w*h),output(w*h);
    std::ifstream input(argv[1],std::ios::binary);input.read(reinterpret_cast<char*>(source.data()),source.size()*4);if(!input)return 3;
    std::vector<float> work(nss::lssc_denoise_work_floats(w,h,b,b));
    nss::lssc_denoise_plane(source.data(),w,h,w,num.data(),den.data(),w,b,b,3.f/255.f,work.data(),int(work.size()));
    nss::aggregate_finish(output.data(),num.data(),den.data(),source.data(),w,h,w,w,w);
    std::ofstream result(argv[5],std::ios::binary);result.write(reinterpret_cast<char*>(output.data()),output.size()*4);if(!result)return 4;
    std::printf("{\"attempts\":%u,\"accepted\":%u,\"full_fp64_searches\":%u,\"refined_columns\":%u}\n",mixed_stats.attempts,mixed_stats.accepted,mixed_stats.attempts-mixed_stats.accepted,mixed_stats.refined_columns);
}
