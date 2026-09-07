#pragma once

#include "nss/checked.hpp"
#include <array>
#include <memory>
#include <mutex>
#include <vector>

namespace nss {
enum class ResourceKind : std::size_t { Workspace, Cached, Pinned, Inflight, Output, Framework, Count };
inline constexpr std::size_t resource_kinds = static_cast<std::size_t>(ResourceKind::Count);
struct ResourceSnapshot {
    std::array<std::size_t, resource_kinds> bytes{};
    std::size_t owned = 0, peak = 0, limit = 0;
};
struct ResourceBudget {
    explicit ResourceBudget(std::size_t limit) { state.limit = limit; }
    ResourceSnapshot snapshot() const { std::lock_guard lock(mu); return state; }
    mutable std::mutex mu;
    ResourceSnapshot state;
};

// An account belongs to the allocation, so evicted chunks remain charged while
// another request pins them. All changes share the node budget's mutex.
class ResourceAccount {
public:
    ResourceAccount(std::shared_ptr<ResourceBudget> budget, ResourceKind kind)
        : budget_(std::move(budget)), kind_(kind) {}
    void acquire(std::size_t bytes) {
        if (!budget_) return;
        std::lock_guard lock(budget_->mu);
        auto& state = budget_->state;
        const bool owned = kind_ != ResourceKind::Framework;
        if (owned && bytes > state.limit - state.owned)
            throw std::runtime_error("nss: memory_limit_mb exceeded");
        const auto index = static_cast<std::size_t>(kind_);
        const auto next = checked_add(state.bytes[index], bytes);
        const auto account_next = checked_add(bytes_, bytes);
        state.bytes[index] = next; bytes_ = account_next;
        if (owned) { state.owned += bytes; state.peak = std::max(state.peak, state.owned); }
    }
    void release(std::size_t bytes) noexcept {
        if (!budget_) return;
        std::lock_guard lock(budget_->mu);
        budget_->state.bytes[static_cast<std::size_t>(kind_)] -= bytes;
        if (kind_ != ResourceKind::Framework) budget_->state.owned -= bytes;
        bytes_ -= bytes;
    }
    void retag(ResourceKind kind) noexcept {
        if (!budget_ || kind == kind_) return;
        std::lock_guard lock(budget_->mu);
        // Retagging is only used between owned chunk categories.
        budget_->state.bytes[static_cast<std::size_t>(kind_)] -= bytes_;
        budget_->state.bytes[static_cast<std::size_t>(kind)] += bytes_;
        kind_ = kind;
    }
private:
    std::shared_ptr<ResourceBudget> budget_;
    ResourceKind kind_;
    std::size_t bytes_ = 0;
};

inline std::shared_ptr<ResourceBudget>& current_budget() {
    static thread_local std::shared_ptr<ResourceBudget> value;
    return value;
}
inline std::shared_ptr<ResourceAccount>& current_account() {
    static thread_local std::shared_ptr<ResourceAccount> value;
    return value;
}
inline std::shared_ptr<ResourceAccount> make_resource_account(ResourceKind kind) {
    return current_budget() ? std::make_shared<ResourceAccount>(current_budget(), kind) : nullptr;
}
class ResourceScope {
public:
    explicit ResourceScope(std::shared_ptr<ResourceBudget> budget,
                           std::shared_ptr<ResourceAccount> account = {})
        : old_budget_(current_budget()), old_account_(current_account()) {
        // Construct before installing TLS so allocation failure leaves it intact.
        if (budget && !account) account = std::make_shared<ResourceAccount>(budget, ResourceKind::Inflight);
        current_budget() = std::move(budget); current_account() = std::move(account);
    }
    ~ResourceScope() { current_budget() = std::move(old_budget_); current_account() = std::move(old_account_); }
    ResourceScope(const ResourceScope&) = delete;
private:
    std::shared_ptr<ResourceBudget> old_budget_;
    std::shared_ptr<ResourceAccount> old_account_;
};

template<class T> struct ResourceAllocator {
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    std::shared_ptr<ResourceAccount> account;
    ResourceAllocator() noexcept : account(current_account()) {}
    explicit ResourceAllocator(std::shared_ptr<ResourceAccount> owner) noexcept : account(std::move(owner)) {}
    template<class U> ResourceAllocator(const ResourceAllocator<U>& other) noexcept : account(other.account) {}
    T* allocate(std::size_t count) {
        const auto bytes = checked_mul(count, sizeof(T));
        if (account) account->acquire(bytes);
        try { return std::allocator<T>{}.allocate(count); }
        catch (...) { if (account) account->release(bytes); throw; }
    }
    void deallocate(T* pointer, std::size_t count) noexcept {
        std::allocator<T>{}.deallocate(pointer, count);
        if (account) account->release(count * sizeof(T));
    }
    template<class U> bool operator==(const ResourceAllocator<U>& other) const noexcept { return account == other.account; }
};
template<class T> using ResourceVector = std::vector<T, ResourceAllocator<T>>;
} // namespace nss
