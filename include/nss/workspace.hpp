#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace nss {

class Workspace {
public:
    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    ~Workspace();

    float* get(std::size_t floats);

private:
    struct Buf {
        float* p = nullptr;
        std::size_t n = 0;
    };
    std::shared_mutex mu_;
    std::unordered_map<std::thread::id, Buf> map_;
};

void* aligned64(std::size_t bytes);
void aligned64_free(void* p);

}  // namespace nss
