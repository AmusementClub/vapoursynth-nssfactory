// SPDX-License-Identifier: GPL-2.0-only
// Compute pipelines from the embedded SPIR-V (runtime/shaders.hpp). Kernels
// take their pointers and sizes in one push-constant block (buffer device
// addresses, D24), so there are no descriptor sets. Creation goes through the
// device's pipeline cache, which is written back to disk afterwards.
#pragma once

#include "vulkan/runtime/context.hpp"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace nss_vk {

struct PipelineDesc {
    std::string_view shader;          // variant id
    std::uint32_t push_bytes = 0;     // size of the push-constant block (<= 128)
    // A required subgroup size with full subgroups (D23); 0 = the driver's choice.
    std::uint32_t subgroup_size = 0;
    // Specialization constants 0, 1, ... as 32-bit values.
    std::vector<std::uint32_t> constants;
};

class Pipeline {
public:
    Pipeline() = default;
    Pipeline(std::shared_ptr<Device> device, const PipelineDesc& desc);
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&& other) noexcept { *this = std::move(other); }
    Pipeline& operator=(Pipeline&& other) noexcept;
    ~Pipeline() { reset(); }
    void reset() noexcept;

    // Binds, pushes `push` (push_bytes of it) and dispatches.
    void dispatch(VkCommandBuffer commands, const void* push, std::uint32_t x, std::uint32_t y = 1,
                  std::uint32_t z = 1) const;
    VkPipeline get() const noexcept { return pipeline_; }
    VkPipelineLayout layout() const noexcept { return layout_; }

private:
    std::shared_ptr<Device> device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::uint32_t push_bytes_ = 0;
};

// Makes compute writes visible to compute reads that follow in the same
// command buffer.
void compute_barrier(const Device& device, VkCommandBuffer commands);

}  // namespace nss_vk
