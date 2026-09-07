#pragma once

#include <VapourSynth4.h>

#include <string>
#include <exception>
#include <new>
#include "host/ownership.hpp"
#include "nss/resources.hpp"

namespace nss {

bool is_const_32f(const VSVideoInfo& vi);
bool same_video(const VSVideoInfo& a, const VSVideoInfo& b);
int plane_width(const VSVideoInfo& vi, int plane);
int plane_height(const VSVideoInfo& vi, int plane);
inline void validate_group_planes(const VSVideoInfo& vi, const float* sigma, int block) {
    for (int plane = 0; plane < vi.format.numPlanes; ++plane) {
        if (sigma[plane] < 0.f) throw std::invalid_argument("nss: sigma must be non-negative");
        if (sigma[plane] > 0.f && (plane_width(vi, plane) < block || plane_height(vi, plane) < block))
            throw std::invalid_argument("nss: selected plane is smaller than block_size");
    }
}

// Validate the immutable public argument map before narrowing any numbers or
// acquiring node references. All current numeric arguments use int/float.
// Algorithm-specific ranges are still checked by the individual factories.
bool validate_numeric_args(const VSAPI* vsapi, const VSMap* in, VSMap* out);

template <auto Create>
void VS_CC checked_create(const VSMap* in, VSMap* out, void* user_data, VSCore* core, const VSAPI* vsapi) {
    try {
        if (!validate_numeric_args(vsapi, in, out)) return;
        int error = 0;
        const auto value = vsapi->mapGetInt(in, "memory_limit_mb", 0, &error);
        const auto megabytes = error ? 1024 : value;
        if (megabytes < 1 || megabytes > 1048576)
            throw std::invalid_argument("nss: memory_limit_mb must be in [1, 1048576]");
        auto budget = std::make_shared<ResourceBudget>(checked_mul(static_cast<std::size_t>(megabytes), 1048576));
        ResourceScope resource_scope(budget);
        Create(in, out, user_data, core, vsapi);
    } catch (const std::bad_alloc&) { vsapi->mapSetError(out, "nss: resource allocation failed during creation"); }
      catch (const std::exception& error) { vsapi->mapSetError(out, error.what()); }
      catch (...) { vsapi->mapSetError(out, "nss: unexpected filter creation failure"); }
}

// The map helpers below require the public argument validation above.
float map_float(const VSAPI* vsapi, const VSMap* in, const char* key, float def, int* err = nullptr);
int map_int(const VSAPI* vsapi, const VSMap* in, const char* key, int def, int* err = nullptr);
void map_float_array(const VSAPI* vsapi, const VSMap* in, const char* key, float* out, int n, float def);
void map_int_array(const VSAPI* vsapi, const VSMap* in, const char* key, int* out, int n, int def);
void map_inherit_int(const VSAPI* vsapi, const VSMap* in, const char* key, int* out, int n, int def);

}  // namespace nss
