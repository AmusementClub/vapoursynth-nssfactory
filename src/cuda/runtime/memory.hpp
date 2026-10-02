// SPDX-License-Identifier: GPL-2.0-only
// Owning device / pinned-host allocations charged to the filter's resource
// budget (memory_limit_mb, D14), plus a bump arena for per-frame scratch.
#pragma once

#include "cuda/runtime/error.hpp"
#include "nss/resources.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace nss_cuda {

// Charges `bytes` to `budget` under `kind` for the lifetime of the object.
class BudgetCharge {
public:
    BudgetCharge() = default;
    BudgetCharge(const std::shared_ptr<nss::ResourceBudget>& budget, nss::ResourceKind kind, std::size_t bytes)
        : account_(budget ? std::make_shared<nss::ResourceAccount>(budget, kind) : nullptr), bytes_(bytes) {
        if (account_) account_->acquire(bytes_);
    }
    BudgetCharge(const BudgetCharge&) = delete;
    BudgetCharge& operator=(const BudgetCharge&) = delete;
    BudgetCharge(BudgetCharge&& other) noexcept
        : account_(std::move(other.account_)), bytes_(std::exchange(other.bytes_, 0)) {}
    BudgetCharge& operator=(BudgetCharge&& other) noexcept {
        if (this != &other) {
            release();
            account_ = std::move(other.account_);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }
    ~BudgetCharge() { release(); }

private:
    void release() noexcept {
        if (account_) account_->release(bytes_);
        account_.reset();
        bytes_ = 0;
    }
    std::shared_ptr<nss::ResourceAccount> account_;
    std::size_t bytes_ = 0;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(std::size_t bytes, const std::shared_ptr<nss::ResourceBudget>& budget = nullptr);
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)),
          charge_(std::move(other.charge_)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    ~DeviceBuffer() { reset(); }

    void reset() noexcept;
    void* get() const noexcept { return ptr_; }
    template <class T> T* as() const noexcept { return static_cast<T*>(ptr_); }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    BudgetCharge charge_;
};

class PinnedBuffer {
public:
    PinnedBuffer() = default;
    PinnedBuffer(std::size_t bytes, const std::shared_ptr<nss::ResourceBudget>& budget = nullptr);
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)),
          charge_(std::move(other.charge_)) {}
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;
    ~PinnedBuffer() { reset(); }

    void reset() noexcept;
    void* get() const noexcept { return ptr_; }
    template <class T> T* as() const noexcept { return static_cast<T*>(ptr_); }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    BudgetCharge charge_;
};

// Bump allocator over one DeviceBuffer. Filters size it at creation from the
// clip geometry (so budget failures happen at creation) and reset it per frame.
class DeviceArena {
public:
    static constexpr std::size_t kAlignment = 256;
    static std::size_t round_up(std::size_t bytes) noexcept { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

    DeviceArena() = default;
    DeviceArena(std::size_t bytes, const std::shared_ptr<nss::ResourceBudget>& budget)
        : buffer_(round_up(bytes), budget) {}

    template <class T> T* take(std::size_t count) {
        const std::size_t bytes = round_up(count * sizeof(T));
        if (bytes > buffer_.bytes() - used_) {
            throw std::logic_error("nss_cuda: device arena exhausted (workspace sizing bug)");
        }
        T* out = reinterpret_cast<T*>(static_cast<std::uint8_t*>(buffer_.get()) + used_);
        used_ += bytes;
        return out;
    }
    void reset() noexcept { used_ = 0; }
    std::size_t capacity() const noexcept { return buffer_.bytes(); }
    std::size_t used() const noexcept { return used_; }

private:
    DeviceBuffer buffer_;
    std::size_t used_ = 0;
};

}  // namespace nss_cuda
