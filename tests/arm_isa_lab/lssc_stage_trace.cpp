#include "nss/cpu_api.hpp"
#include "nss/cpu_lssc.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
namespace {
std::filesystem::path directory;
std::ofstream trace;
int frame_id=0,call_id=-1;
template<class T> void save(std::string name,const T* p,int n) {
 std::ofstream out(directory/name,std::ios::binary);out.write(reinterpret_cast<const char*>(p),sizeof(T)*n);if(!out)throw std::runtime_error("write failed");
}
template<class T> void list(const T* p,int n) {trace<<'[';for(int i=0;i<n;++i){if(i)trace<<',';trace<<+p[i];}trace<<']';}
}
void lssc_step(const float* y,int m,const float* D,int atoms,int ldd,const float* work,const unsigned char* used,int selected,double magnitude,int step) {
 if(step==0){++call_id;save("call"+std::to_string(call_id)+"-D.f32",D,ldd*atoms);save("call"+std::to_string(call_id)+"-y.f32",y,m);}
 std::vector<double> residual(m);for(int i=0;i<m;++i)std::memcpy(&residual[i],work+2*i,sizeof(double));
 std::vector<long double> oracle(atoms);long double largest=0,second=0;int best=-1;
 for(int a=0;a<atoms;++a){for(int i=0;i<m;++i)oracle[a]+=(long double)D[a*ldd+i]*residual[i];if(used[a])continue;long double v=std::abs(oracle[a]);if(v>largest){second=largest;largest=v;best=a;}else if(v>second)second=v;}
 trace<<"{\"kind\":\"step\",\"frame\":"<<frame_id<<",\"call\":"<<call_id<<",\"m\":"<<m<<",\"atoms\":"<<atoms<<",\"ldd\":"<<ldd<<",\"step\":"<<step<<",\"selected\":"<<selected<<",\"magnitude\":"<<magnitude<<",\"oracle_selected\":"<<best<<",\"oracle_margin\":"<<largest-second<<",\"residual\":";list(residual.data(),m);trace<<",\"oracle\":";list(oracle.data(),atoms);trace<<"}\n";
}
void lssc_coefficients(const float* a,int atoms,const int* support,int count){trace<<"{\"kind\":\"coefficients\",\"frame\":"<<frame_id<<",\"call\":"<<call_id<<",\"support\":";list(support,count);trace<<",\"coefficients\":";list(a,atoms);trace<<"}\n";}
void lssc_dictionary(float* D,int m,int atoms,int ldd,int stage){
 const char* key=stage==0?"NSS_LSSC_INITIAL_DICT":"NSS_LSSC_FINAL_DICT";
 const std::string name="frame"+std::to_string(frame_id)+(stage==0?"-initial-D.f32":"-final-D.f32");
 if(const char* path=std::getenv(key)){std::ifstream in(std::filesystem::path(path)/name,std::ios::binary);in.read(reinterpret_cast<char*>(D),sizeof(float)*ldd*atoms);if(!in)throw std::runtime_error("dictionary replay failed");}
 save(name,D,ldd*atoms);
}
int main(int argc,char**argv){
 if(argc!=3)return 2;directory=argv[1];if(!std::filesystem::create_directories(directory))return 3;int block=std::atoi(argv[2]);
 trace.open(directory/"trace.jsonl");trace<<std::setprecision(21);const int w=36,h=34,m=block*block;
 for(frame_id=0;frame_id<3;++frame_id){std::vector<float> source(w*h),num(w*h),den(w*h),result(w*h);for(int y=0;y<h;++y)for(int x=0;x<w;++x)source[y*w+x]=float((x*3+y*5+frame_id*7)%97)/128.f;
 std::vector<float> work(nss::lssc_denoise_work_floats(w,h,block,block));nss::lssc_denoise_plane(source.data(),w,h,w,num.data(),den.data(),w,block,block,3.f/255.f,work.data(),int(work.size()));nss::aggregate_finish(result.data(),num.data(),den.data(),source.data(),w,h,w,w,w);
 save("frame"+std::to_string(frame_id)+"-output.f32",result.data(),w*h);save("frame"+std::to_string(frame_id)+"-den.f32",den.data(),w*h);}
 return trace.good()?0:4;
}
