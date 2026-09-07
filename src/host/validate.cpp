#include "host/validate.hpp"

#include <VSHelper4.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace nss {

bool validate_numeric_args(const VSAPI* vsapi, const VSMap* in, VSMap* out) {
    const int keys = vsapi->mapNumKeys(in);
    for (int i = 0; i < keys; ++i) {
        const char* key = vsapi->mapGetKey(in, i);
        const int type = vsapi->mapGetType(in, key);
        if (type != ptInt && type != ptFloat) {
            continue;
        }
        const int count = vsapi->mapNumElements(in, key);
        for (int index = 0; index < count; ++index) {
            int error = 0;
            bool valid;
            if (type == ptInt) {
                const int64_t value = vsapi->mapGetInt(in, key, index, &error);
                valid = value >= std::numeric_limits<int>::min() && value <= std::numeric_limits<int>::max();
            } else {
                const double value = vsapi->mapGetFloat(in, key, index, &error);
                const bool finite = (std::bit_cast<std::uint64_t>(value) & UINT64_C(0x7ff0000000000000)) !=
                                    UINT64_C(0x7ff0000000000000);
                const double limit = std::numeric_limits<float>::max();
                valid = finite && value >= -limit && value <= limit;
            }
            if (error || !valid) {
                char message[256];
                std::snprintf(message, sizeof(message),
                              "nss: parameter %s[%d] is not representable as a finite %s",
                              key, index, type == ptInt ? "int" : "float");
                vsapi->mapSetError(out, message);
                return false;
            }
        }
    }
    return true;
}

bool is_const_32f(const VSVideoInfo& vi) {
    if (vi.width < 1 || vi.height < 1 || vi.numFrames < 1 ||
        static_cast<std::uint64_t>(vi.width) * vi.height > std::numeric_limits<int>::max() ||
        !vsh::isConstantVideoFormat(&vi)) {
        return false;
    }
    return vi.format.sampleType == stFloat && vi.format.bitsPerSample == 32 &&
           (vi.format.colorFamily == cfGray || vi.format.colorFamily == cfYUV ||
            vi.format.colorFamily == cfRGB);
}

bool same_video(const VSVideoInfo& a, const VSVideoInfo& b) {
    return vsh::isSameVideoInfo(&a, &b) && a.numFrames == b.numFrames;
}

int plane_width(const VSVideoInfo& vi, int plane) {
    return vi.width >> (plane ? vi.format.subSamplingW : 0);
}

int plane_height(const VSVideoInfo& vi, int plane) {
    return vi.height >> (plane ? vi.format.subSamplingH : 0);
}

float map_float(const VSAPI* vsapi, const VSMap* in, const char* key, float def, int* err_out) {
    int err = 0;
    const double v = vsapi->mapGetFloat(in, key, 0, &err);
    if (err_out) {
        *err_out = err;
    }
    return err ? def : static_cast<float>(v);
}

int map_int(const VSAPI* vsapi, const VSMap* in, const char* key, int def, int* err_out) {
    int err = 0;
    const int64_t v = vsapi->mapGetInt(in, key, 0, &err);
    if (err_out) {
        *err_out = err;
    }
    return err ? def : static_cast<int>(v);
}

void map_float_array(const VSAPI* vsapi, const VSMap* in, const char* key, float* out, int n, float def) {
    int err = 0;
    const int m = vsapi->mapNumElements(in, key);
    if (m <= 0) {
        for (int i = 0; i < n; ++i) {
            out[i] = def;
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        err = 0;
        const int idx = (i < m) ? i : (m - 1);
        out[i] = static_cast<float>(vsapi->mapGetFloat(in, key, idx, &err));
        if (err) {
            out[i] = def;
        }
    }
}

void map_int_array(const VSAPI* vsapi, const VSMap* in, const char* key, int* out, int n, int def) {
    int err = 0;
    const int m = vsapi->mapNumElements(in, key);
    if (m <= 0) {
        for (int i = 0; i < n; ++i) {
            out[i] = def;
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        err = 0;
        const int idx = (i < m) ? i : (m - 1);
        out[i] = static_cast<int>(vsapi->mapGetInt(in, key, idx, &err));
        if (err) {
            out[i] = def;
        }
    }
}

void map_inherit_int(const VSAPI* vsapi, const VSMap* in, const char* key, int* out, int n, int def) {
    const int m = vsapi->mapNumElements(in, key);
    if (m <= 0) {
        for (int i = 0; i < n; ++i) {
            out[i] = def;
        }
        return;
    }
    for (int i = 0; i < n; ++i) {
        int err = 0;
        if (i < m) {
            out[i] = static_cast<int>(vsapi->mapGetInt(in, key, i, &err));
            if (err) {
                out[i] = (i == 0) ? def : out[i - 1];
            }
        } else {
            out[i] = out[i - 1];
        }
    }
}

}  // namespace nss
