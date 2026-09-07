#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include "nss/resources.hpp"

namespace nss {

class Workspace {
public:
    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace();

    // Per-thread scratch, aligned to 64 bytes. Contents are unspecified after
    // successful growth. Overflow/allocation failure throws std::bad_alloc (or
    // a derived exception) and preserves the old pointer, capacity and contents.
    // get(0) returns a stable non-null allocation; destruction frees all buffers.
    float* get(std::size_t floats);
    // Call before the first request; the caller must serialize all get/use
    // operations for a serial workspace (rolling's compute_mu does this).
    void set_serial();
    std::size_t buffer_count() const;


private:
    struct Buf {
        float* p = nullptr;
        std::size_t n = 0;
    };
    std::shared_ptr<ResourceAccount> account_ = make_resource_account(ResourceKind::Workspace);
    bool serial_ = false;
    mutable std::shared_mutex mu_;
    using MapValue = std::pair<const std::thread::id, Buf>;
    std::unordered_map<std::thread::id, Buf, std::hash<std::thread::id>, std::equal_to<std::thread::id>,
                       ResourceAllocator<MapValue>> map_{0, std::hash<std::thread::id>{},
                       std::equal_to<std::thread::id>{}, ResourceAllocator<MapValue>{account_}};
};

void* aligned64(std::size_t bytes);
void aligned64_free(void* p);

}  // namespace nss
