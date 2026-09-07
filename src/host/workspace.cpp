#include "nss/workspace.hpp"

#include <cstdlib>
#include <cstring>
#include <limits>
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
        if (account_ && kv.second.p) account_->release(std::max<std::size_t>(64, kv.second.n * sizeof(float)));
    }
}

float* Workspace::get(std::size_t floats) {
    if (floats > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::bad_array_new_length();
    }
    const std::size_t bytes = std::max<std::size_t>(64, floats * sizeof(float));
    const auto id = serial_ ? std::thread::id{} : std::this_thread::get_id();
    {
        std::shared_lock lock(mu_);
        auto it = map_.find(id);
        if (it != map_.end() && it->second.p && it->second.n >= floats) {
            return it->second.p;
        }
    }
    std::unique_lock lock(mu_);
    auto& buf = map_[id];
    if (buf.n >= floats && buf.p) {
        return buf.p;
    }
    if (account_) account_->acquire(bytes);
    float* replacement = static_cast<float*>(aligned64(bytes));
    if (!replacement) {
        if (account_) account_->release(bytes);
        throw std::bad_alloc();
    }
    aligned64_free(buf.p);
    if (account_ && buf.p) account_->release(std::max<std::size_t>(64, buf.n * sizeof(float)));
    buf.p = replacement;
    buf.n = floats;
    return buf.p;
}

void Workspace::set_serial() {
    std::unique_lock lock(mu_);
    if (!map_.empty()) throw std::logic_error("nss: workspace mode already in use");
    serial_ = true;
}
std::size_t Workspace::buffer_count() const {
    std::shared_lock lock(mu_);
    return map_.size();
}
}  // namespace nss
