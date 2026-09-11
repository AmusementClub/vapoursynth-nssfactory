// Test-only ELF interposition for the private matcher recorder's input hashes.
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <vector>
std::uint64_t nss_trace_hash(const float* p,int width,int height,int stride) {
 if(!p||width<0||height<0||stride<width)return 0;
 std::uint64_t hash=14695981039346656037ull;std::vector<float> data;
 for(int y=0;y<height;++y)for(int x=0;x<width;++x){std::uint32_t bits;float value=p[y*stride+x];std::memcpy(&bits,&value,4);hash^=bits;hash*=1099511628211ull;data.push_back(value);}
 if(const char* out=std::getenv("NSS_MATCH_HASH_DUMP_DIR")){
  static std::mutex mutex;std::lock_guard<std::mutex> lock(mutex);std::filesystem::path root(out);std::filesystem::create_directories(root);std::string name=std::to_string(hash);
  if(!std::filesystem::exists(root/(name+".f32"))){std::ofstream f(root/(name+".f32"),std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*4);std::ofstream j(root/(name+".json"));j<<"{\"width\":"<<width<<",\"height\":"<<height<<",\"stride\":"<<stride<<"}\n";if(!f||!j)throw std::runtime_error("hash input capture failed");}
 }
 return hash;
}
