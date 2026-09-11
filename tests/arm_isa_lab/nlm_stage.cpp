#include "nss/cpu_api.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

void save(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float));
    if (!file) std::abort();
}
int main(int argc, char** argv) {
    if (argc != 8) return 2;
    const int w=std::atoi(argv[2]),h=std::atoi(argv[3]),a=std::atoi(argv[4]),s=std::atoi(argv[5]);
    const float strength=std::atof(argv[6]); const std::filesystem::path out(argv[7]);
    if (!std::filesystem::create_directory(out)) return 3;
    const int size=w*h;
    std::vector<float> src(size), horizontal(size), weights(size), weight(size), numerator(size), maximum(size,std::numeric_limits<float>::epsilon()), result(size), scratch(size), buffer(w);
    std::ifstream input(argv[1],std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(src.data()),size*sizeof(float))) return 4;
    const float scale=(255.f*255.f)/(3.f*strength*strength*float(2*s+1)*float(2*s+1));
    save(out/"scale.f32",std::vector<float>{scale});
    int index=0;
    for(int dy=-a;dy<=a;++dy)for(int dx=-a;dx<=a;++dx) {
        if (dy*(2*a+1)+dx>=0) continue;
        nss::nlm_distance_luma_horizontal_f32(horizontal.data(),scratch.data(),src.data(),src.data(),dx,dy,s,w,h,w);
        nss::nlm_vertical_welsch(weights.data(),horizontal.data(),s,scale,w,h,w,buffer.data());
        save(out/("h"+std::to_string(index)+".f32"),horizontal);
        save(out/("w"+std::to_string(index)+".f32"),weights);
        nss::nlm_accum_ch1(weight.data(),numerator.data(),maximum.data(),src.data(),src.data(),weights.data(),weights.data(),dx,dy,w,h,w);
        ++index;
    }
    nss::nlm_finish_ch1(result.data(),src.data(),weight.data(),numerator.data(),maximum.data(),1.f,w,h,w);
    save(out/"output.f32",result);save(out/"weight.f32",weight);save(out/"numerator.f32",numerator);save(out/"maximum.f32",maximum);
}
