#include "nss/workspace.hpp"

#include <cstdlib>
#include <cstring>
#include <new>

namespace nss {

void* aligned64(std::size_t bytes) {
    if (bytes == 0) {
        bytes = 64;
    }
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes) != 0) {
        return nullptr;
    }
    return p;
}

void aligned64_free(void* p) {
    std::free(p);
}

Workspace::~Workspace() {
    for (auto& kv : map_) {
        aligned64_free(kv.second.p);
    }
}

float* Workspace::get(std::size_t floats) {
    const auto id = std::this_thread::get_id();
    {
        std::shared_lock lock(mu_);
        auto it = map_.find(id);
        if (it != map_.end() && it->second.n >= floats) {
            return it->second.p;
        }
    }
    std::unique_lock lock(mu_);
    auto& buf = map_[id];
    if (buf.n >= floats && buf.p) {
        return buf.p;
    }
    aligned64_free(buf.p);
    buf.p = static_cast<float*>(aligned64(floats * sizeof(float)));
    if (!buf.p) {
        buf.n = 0;
        throw std::bad_alloc();
    }
    buf.n = floats;
    return buf.p;
}

}  // namespace nss
