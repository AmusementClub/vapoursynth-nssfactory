// SPDX-License-Identifier: GPL-2.0-only
#include "vulkan/runtime/context.hpp"

#include "vulkan/runtime/error.hpp"
#include "vulkan/runtime/instance.hpp"
#include "vulkan/runtime/shaders.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <random>
#include <system_error>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace nss_vk {
namespace {

std::string prefix(const char* filter) {
    return std::string("nss_vk.") + filter + ": ";
}

const char* env(const char* name) {
    const char* value = std::getenv(name);
    return value && *value ? value : nullptr;
}

// A path from the environment; wide on Windows, where the narrow variables
// are in the ANSI code page and lose other characters.
std::filesystem::path env_path(const char* name) {
#ifdef _WIN32
    const std::wstring wide(name, name + std::strlen(name));
    const wchar_t* value = _wgetenv(wide.c_str());
    return value && *value ? std::filesystem::path(value) : std::filesystem::path();
#else
    const char* value = env(name);
    return value ? std::filesystem::path(value) : std::filesystem::path();
#endif
}

std::uint64_t process_id() {
#ifdef _WIN32
    return static_cast<std::uint64_t>(_getpid());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

// The legacy BAR window: device-local, host-visible heaps of this size or less
// are not resizable BAR, and uploads go through staging instead.
constexpr std::uint64_t kLegacyBarBytes = 256ull << 20;

}  // namespace

std::uint64_t wait_timeout_ns() {
    static const std::uint64_t ns = [] {
        std::uint64_t ms = kDefaultWaitTimeoutMs;
        if (const char* value = env("NSS_VK_WAIT_TIMEOUT_MS")) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (end && *end == '\0' && parsed > 0) ms = std::min<unsigned long long>(parsed, 86'400'000ull);
        }
        return ms * 1'000'000ull;
    }();
    return ns;
}

std::string signature(const char* shared) {
    return std::string(shared) + kBackendSignature;
}

BackendArgs parse_backend_args(const VSAPI* vsapi, const VSMap* in, const char* filter) {
    BackendArgs args;
    int err = 0;
    const int64_t device = vsapi->mapGetInt(in, "device_id", 0, &err);
    if (!err) {
        if (device < 0 || device > INT32_MAX) {
            throw std::invalid_argument(prefix(filter) + "device_id must be in [0, " + std::to_string(INT32_MAX) + "]");
        }
        args.device_id = static_cast<int>(device);
    }
    const int64_t streams = vsapi->mapGetInt(in, "num_streams", 0, &err);
    if (!err) {
        if (streams < 1 || streams > kMaxStreams) {
            throw std::invalid_argument(prefix(filter) + "num_streams must be in [1, " + std::to_string(kMaxStreams) + "]");
        }
        args.num_streams = static_cast<int>(streams);
        args.streams_explicit = true;
    }
    return args;
}

std::filesystem::path pipeline_cache_directory() {
    if (const std::filesystem::path value = env_path("NSS_VK_PIPELINE_CACHE"); !value.empty()) {
        if (value == "0" || value == "off") return {};
        return value;
    }
#ifdef _WIN32
    if (const auto local = env_path("LOCALAPPDATA"); !local.empty()) return local / "nssfactory" / "vk";
#else
    if (const auto cache = env_path("XDG_CACHE_HOME"); !cache.empty()) return cache / "nssfactory" / "vk";
    if (const auto home = env_path("HOME"); !home.empty()) return home / ".cache" / "nssfactory" / "vk";
#endif
    return {};
}

std::shared_ptr<Device> Device::open(int index, bool allow_unsupported) {
    const Instance& vk = instance();
    if (vk.devices.empty()) {
        throw std::runtime_error("no Vulkan device available (" + vk.error + ")");
    }
    if (index < 0 || index >= static_cast<int>(vk.devices.size())) {
        throw std::runtime_error("device_id " + std::to_string(index) + " out of range (" +
                                 std::to_string(vk.devices.size()) + " Vulkan devices)");
    }
    // One logical device per physical device for the life of the process, as
    // CUDA's primary context: filters come and go, the device stays, and
    // creating a filter never pays for vkCreateDevice again (owner,
    // 2026-10-24). The registry is never destroyed: no Vulkan calls at exit
    // or plugin unload, where the loader may already be gone. The price is
    // one leaked device per plugin reload when VapourSynth unloads plugins
    // with their cores (unless disableLibraryUnloading).
    static auto* mutex = new std::mutex();
    static auto* devices = new std::map<int, std::shared_ptr<Device>>();
    // A poisoned device (a wait timed out or the device was lost) leaks
    // everything freed on it: it is set aside here, never destroyed, and the
    // next filter gets a fresh device.
    static auto* poisoned = new std::vector<std::shared_ptr<Device>>();
    std::lock_guard lock(*mutex);
    std::shared_ptr<Device> slot = (*devices)[index];
    if (slot && slot->poisoned()) {
        poisoned->push_back(std::move(slot));
        slot.reset();
    }
    if (!slot) {
        std::shared_ptr<Device> device(new Device());
        device->physical_ = vk.devices[static_cast<std::size_t>(index)];
        device->caps_ = query_caps(device->physical_);
        device->support_ = classify(device->caps_);
        if (device->support_.support == Support::unsupported && !allow_unsupported) {
            throw std::runtime_error(device->label() + " is not supported: missing " + join(device->support_.missing));
        }
        device->create(index);  // on failure ~Device frees what was created
        (*devices)[index] = device;
        slot = std::move(device);
    } else if (slot->support_.support == Support::unsupported && !allow_unsupported) {
        throw std::runtime_error(slot->label() + " is not supported: missing " + join(slot->support_.missing));
    }
    return slot;
}

Device::~Device() {
    // A failed wait may have left work running on objects that are gone with
    // the device: leak it instead (Device::poison).
    // Otherwise nothing runs: every Slot holds the device and settles (waits,
    // bounded) before it releases it.
    if (poisoned()) return;
    if (pipeline_cache_) table_.vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
    if (allocator_) vmaDestroyAllocator(allocator_);
    if (device_ != VK_NULL_HANDLE) table_.vkDestroyDevice(device_, nullptr);
}

std::string Device::label() const {
    return caps_.name + " (" + caps_.driver_name + " " + caps_.driver_info + ")";
}

void Device::create(int index) {
    index_ = index;
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
    families.resize(family_count);

    // Compute: the first family with compute (the queue the V0 spikes measured).
    std::uint32_t compute_family = family_count;
    for (std::uint32_t f = 0; f < family_count && compute_family == family_count; ++f) {
        if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) compute_family = f;
    }
    if (compute_family == family_count) throw std::runtime_error(label() + ": no compute queue");
    // Copy queues: up to two queues of copy-only families (D28).
    // NSS_VK_COPY_QUEUES=0 or 1 uses fewer (A/B measurements; GPU-AV in SDK
    // 1.4.363 fails on copy-only queues and then crashes, so GPU-AV runs use 0).
    std::size_t copy_limit = 2;
    if (const char* value = env("NSS_VK_COPY_QUEUES")) {
        if (value[0] >= '0' && value[0] <= '2' && value[1] == '\0') copy_limit = static_cast<std::size_t>(value[0] - '0');
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> copies;  // (family, index)
    for (std::uint32_t f = 0; f < family_count && copies.size() < copy_limit; ++f) {
        const VkQueueFlags flags = families[f].queueFlags & ~(VK_QUEUE_SPARSE_BINDING_BIT | VK_QUEUE_PROTECTED_BIT);
        if (flags != VK_QUEUE_TRANSFER_BIT) continue;
        for (std::uint32_t q = 0; q < families[f].queueCount && copies.size() < copy_limit; ++q) copies.emplace_back(f, q);
    }
    std::map<std::uint32_t, std::uint32_t> queues_per_family{{compute_family, 1u}};
    for (const auto& [family, queue] : copies) {
        queues_per_family[family] = std::max(queues_per_family[family], queue + 1);
    }
    const float priorities[2] = {1.0f, 1.0f};
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    for (const auto& [family, count] : queues_per_family) {
        VkDeviceQueueCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        info.queueFamilyIndex = family;
        info.queueCount = count;
        info.pQueuePriorities = priorities;
        queue_infos.push_back(info);
    }

    std::uint32_t extension_count = 0;
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &extension_count, nullptr);
    std::vector<VkExtensionProperties> available(extension_count);
    vkEnumerateDeviceExtensionProperties(physical_, nullptr, &extension_count, available.data());
    available.resize(extension_count);
    std::vector<const char*> extensions;
    bool memory_budget = false;
    for (const VkExtensionProperties& e : available) {
        if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
            memory_budget = true;
            extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
        // Imports of VapourSynth frame memory (D28 as amended 2026-10-25).
        if (std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0 && caps_.import_alignment > 0) {
            host_import_ = true;
            extensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        }
    }

    // D22 requirements; synchronization2 is core in 1.3.
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    f12.shaderBufferInt64Atomics = caps_.buffer_int64_atomics;
    f12.hostQueryReset = caps_.host_query_reset;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &f12};
    f13.subgroupSizeControl = caps_.subgroup_size_control;
    f13.computeFullSubgroups = caps_.compute_full_subgroups;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f13};
    features.features.shaderInt64 = caps_.shader_int64;
    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
    info.queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size());
    info.pQueueCreateInfos = queue_infos.data();
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    // Output parameters are undefined on failure: only a created device is kept.
    VkDevice created = VK_NULL_HANDLE;
    NSS_VK_CHECK(vkCreateDevice(physical_, &info, nullptr, &created));
    device_ = created;
    volkLoadDeviceTable(&table_, device_);

    // One mutex per distinct queue.
    queue_mutexes_ = std::make_unique<std::mutex[]>(1 + copies.size());
    auto make_queue = [&](std::uint32_t family, std::uint32_t queue, std::size_t mutex) {
        Queue q;
        table_.vkGetDeviceQueue(device_, family, queue, &q.handle);
        q.family = family;
        q.index = queue;
        q.timestamp_bits = families[family].timestampValidBits;
        q.mutex = &queue_mutexes_[mutex];
        return q;
    };
    compute_ = make_queue(compute_family, 0, 0);
    upload_ = copies.empty() ? compute_ : make_queue(copies[0].first, copies[0].second, 1);
    if (copies.size() >= 2) {
        download_ = make_queue(copies[1].first, copies[1].second, 2);
    } else {
        // One copy queue (D28 as amended: uploads run ahead). Queues run their
        // batches in order, so a download batch waiting there for its kernel
        // would hold every later upload. Default: both directions on the copy
        // queue, downloads submitted only once their kernel is done (Slot).
        // NSS_VK_ONE_COPY_QUEUE=compute puts downloads behind the kernels on
        // the compute queue instead, =shared submits them at once (the layout
        // before; diagnostics for A/B measurements).
        const char* one = env("NSS_VK_ONE_COPY_QUEUE");
        const bool compute = one && std::strcmp(one, "compute") == 0;
        download_ = compute ? compute_ : upload_;
        defer_downloads_ = !copies.empty() && !compute && !(one && std::strcmp(one, "shared") == 0);
    }
    for (const Queue* q : {&compute_, &upload_, &download_}) {
        if (std::find(families_.begin(), families_.end(), q->family) == families_.end()) families_.push_back(q->family);
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_, &properties);
    timestamp_period_ = properties.limits.timestampPeriod;
    debug_labels_ = instance().debug_utils && vkCmdBeginDebugUtilsLabelEXT != nullptr;
    const char* rebar = env("NSS_VK_REBAR");
    rebar_ = caps_.host_visible_device_bytes > kLegacyBarBytes && !(rebar && std::strcmp(rebar, "0") == 0);

    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                           (memory_budget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0);
    allocator_info.physicalDevice = physical_;
    allocator_info.device = device_;
    allocator_info.instance = instance().handle;
    allocator_info.vulkanApiVersion = kRequiredApiVersion;
    allocator_info.pVulkanFunctions = &functions;
    // Smaller blocks than VMA's 256 MiB default: VMA keeps an empty block per
    // memory type, which memory_limit_mb does not see. Plane-sized buffers
    // get dedicated allocations anyway (Buffer).
    allocator_info.preferredLargeHeapBlockSize = 32ull << 20;
    NSS_VK_CHECK(vmaCreateAllocator(&allocator_info, &allocator_));

    load_pipeline_cache();
}

void Device::submit(const Queue& queue, const VkSubmitInfo2& info) const {
    std::lock_guard lock(*queue.mutex);
    NSS_VK_CHECK(table_.vkQueueSubmit2(queue.handle, 1, &info, VK_NULL_HANDLE));
}

void Device::load_pipeline_cache() {
    std::vector<char> data;
    const std::filesystem::path directory = pipeline_cache_directory();
    if (!directory.empty()) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_, &properties);
        char uuid[2 * VK_UUID_SIZE + 1] = {};
        for (int i = 0; i < VK_UUID_SIZE; ++i) std::snprintf(uuid + 2 * i, 3, "%02x", properties.pipelineCacheUUID[i]);
        char name[160];
        std::snprintf(name, sizeof(name), "%04x-%04x-%08x-%s-%016llx.bin", properties.vendorID, properties.deviceID,
                      properties.driverVersion, uuid, static_cast<unsigned long long>(shader_set_hash()));
        cache_file_ = directory / name;
        std::ifstream file(cache_file_, std::ios::binary);
        if (file) data.assign(std::istreambuf_iterator<char>(file), {});
        // The driver checks the header too; drop anything that is not this
        // device's cache rather than hand it over.
        VkPipelineCacheHeaderVersionOne header{};
        if (data.size() >= sizeof(header)) std::memcpy(&header, data.data(), sizeof(header));
        if (data.size() < sizeof(header) || header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
            header.vendorID != properties.vendorID || header.deviceID != properties.deviceID ||
            std::memcmp(header.pipelineCacheUUID, properties.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
            data.clear();
        }
    }
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = data.size();
    info.pInitialData = data.empty() ? nullptr : data.data();
    if (table_.vkCreatePipelineCache(device_, &info, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        info.initialDataSize = 0;  // a rejected file: start empty
        info.pInitialData = nullptr;
        NSS_VK_CHECK(table_.vkCreatePipelineCache(device_, &info, nullptr, &pipeline_cache_));
        data.clear();
    }
    cache_saved_hash_ = 0;
}

void Device::save_pipeline_cache() {
    if (cache_file_.empty()) return;
    std::lock_guard lock(cache_mutex_);
    std::size_t size = 0;
    if (table_.vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr) != VK_SUCCESS || size == 0) return;
    std::vector<char> data(size);
    if (table_.vkGetPipelineCacheData(device_, pipeline_cache_, &size, data.data()) < VK_SUCCESS) return;
    data.resize(size);
    std::uint64_t hash = 1469598103934665603ull;
    for (const char c : data) hash = (hash ^ static_cast<unsigned char>(c)) * 1099511628211ull;
    if (hash == cache_saved_hash_) return;
    // Write a temporary file and rename it over the cache, so a concurrent
    // reader in another process never sees a partial file.
    std::error_code error;
    std::filesystem::create_directories(cache_file_.parent_path(), error);
    std::filesystem::path temporary = cache_file_;
    // Unique per process and call: two processes may save the same cache.
    temporary += ".tmp" + std::to_string(process_id()) + "-" + std::to_string(std::random_device{}());
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file || !file.write(data.data(), static_cast<std::streamsize>(data.size()))) {
            file.close();
            std::filesystem::remove(temporary, error);
            return;
        }
    }
    std::filesystem::rename(temporary, cache_file_, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return;
    }
    cache_saved_hash_ = hash;
}

std::shared_ptr<Device> acquire_device(int device_id, const char* filter, VSCore* core, const VSAPI* vsapi) {
    std::shared_ptr<Device> device;
    try {
        device = Device::open(device_id);
    } catch (const std::exception& error) {
        throw std::invalid_argument(prefix(filter) + error.what());
    }
    if (device->support().support == Support::emulated) {
        static std::mutex mutex;
        static std::set<int> warned;
        std::lock_guard lock(mutex);
        if (warned.insert(device->index()).second) {
            const std::string message = "nss_vk: " + device->label() + " has subgroups of at most " +
                                        std::to_string(device->support().native_lanes) +
                                        " lanes; wider kernels run on the emulated warp layer (same results, slower)";
            vsapi->logMessage(mtWarning, message.c_str(), core);
        }
    }
    return device;
}

}  // namespace nss_vk
