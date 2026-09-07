// Compile the actual BM3D callback against a counted VS provider.
#include "../src/host/filter_bm3d.cpp"
#include <cerrno>
#include <cstdio>
#include <map>
#include <string>

namespace {
int new_failure = 0, new_calls = 0;
bool count_new = false;
}
void* operator new(std::size_t size) {
    if (count_new && ++new_calls == new_failure) throw std::bad_alloc();
    if (void* pointer = std::malloc(size ? size : 1)) return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

struct VSMap { std::map<std::string, int64_t> values; };
struct VSFrame {
    int width = 16, height = 16, refs = 0;
    ptrdiff_t stride_bytes = 0;
    bool output = false;
    std::vector<float> pixels;
    VSMap props;
};
namespace {
VSVideoFormat format{cfGray, stFloat, 32, 4, 0, 0, 1};
VSFrame input;
int acquired = 0, live_outputs = 0, calls = 0, fail_at = 0, errors = 0;
bool fail_workspace = false;
const VSFrame* VS_CC get_frame(int, VSNode*, VSFrameContext*) noexcept {
    if (++calls == fail_at) return nullptr;
    ++input.refs; ++acquired; return &input;
}
VSFrame* VS_CC new_frame(const VSVideoFormat*, int w, int h, const VSFrame*, VSCore*) noexcept {
    if (++calls == fail_at) return nullptr;
    try {
    auto frame = std::make_unique<VSFrame>();
    frame->width = w; frame->height = h; frame->refs = 1; frame->output = true;
    frame->pixels.resize(static_cast<std::size_t>(w) * h);
    ++live_outputs; return frame.release();
    } catch (...) { return nullptr; }
}
void VS_CC free_frame(const VSFrame* f) noexcept {
    auto* frame = const_cast<VSFrame*>(f);
    if (--frame->refs < 0) std::abort();
    if (frame->output && !frame->refs) { --live_outputs; delete frame; }
}
int VS_CC width(const VSFrame* frame, int) noexcept { return frame->width; }
int VS_CC height(const VSFrame* frame, int) noexcept { return frame->height; }
ptrdiff_t VS_CC stride(const VSFrame* frame, int) noexcept {
    return frame->stride_bytes ? frame->stride_bytes : frame->width * sizeof(float);
}
const uint8_t* VS_CC read_ptr(const VSFrame* frame, int) noexcept { return reinterpret_cast<const uint8_t*>(frame->pixels.data()); }
uint8_t* VS_CC write_ptr(VSFrame* frame, int) noexcept { return reinterpret_cast<uint8_t*>(frame->pixels.data()); }
const VSVideoFormat* VS_CC frame_format(const VSFrame*) noexcept { return &format; }
VSMap* VS_CC props(VSFrame* frame) noexcept { return &frame->props; }
int VS_CC set_int(VSMap* map, const char* key, int64_t value, int) noexcept { try { map->values[key] = value; return 0; } catch (...) { return 1; } }
int VS_CC set_int_array(VSMap*, const char*, const int64_t*, int) noexcept { return 0; }
int VS_CC delete_key(VSMap* map, const char* key) noexcept { return static_cast<int>(map->values.erase(key)); }
void VS_CC set_error(const char*, VSFrameContext*) noexcept { ++errors; }
VSAPI make_api() {
    VSAPI api{};
    api.getFrameFilter = get_frame; api.newVideoFrame = new_frame; api.freeFrame = free_frame;
    api.getFrameWidth = width; api.getFrameHeight = height; api.getStride = stride;
    api.getReadPtr = read_ptr; api.getWritePtr = write_ptr; api.getVideoFrameFormat = frame_format;
    api.getFramePropertiesRW = props; api.mapSetInt = set_int; api.mapSetIntArray = set_int_array;
    api.mapDeleteKey = delete_key; api.setFilterError = set_error;
    return api;
}
bool run(int failure, bool workspace_failure, std::size_t limit, bool expect_error, int allocation_failure = 0, bool rolling = false) {
    const auto api = make_api();
    calls = acquired = errors = 0; fail_at = failure; fail_workspace = workspace_failure;
    auto budget = std::make_shared<nss::ResourceBudget>(limit);
    {
        nss::ResourceScope creation(budget);
        RollingData roll;
        auto& data = roll.bm;
        data.vi.format = format; data.vi.width = 16; data.vi.height = 16; data.vi.numFrames = 5;
        data.radius = 1; data.vi_out = data.vi; data.vi_out.height = 96;
        data.sigma[0] = .01f;
        data.block_step[0] = 8; data.bm_range[0] = 1;
        if (rolling) { data.ws.set_serial(); roll.rolling_chunk = 2; roll.cache_limit = 1; }
        new_calls = 0; new_failure = allocation_failure; count_new = true;
        const VSFrame* result = rolling
            ? nss::checked_frame<rollingGetFrame>(2, arAllFramesReady, &roll, nullptr, nullptr, nullptr, &api)
            : nss::checked_frame<bm3dGetFrame>(2, arAllFramesReady, &data, nullptr, nullptr, nullptr, &api);
        count_new = false;
        if (result) api.freeFrame(result);
        if (allocation_failure && result) {
            // std::stable_sort may safely fall back to in-place sorting.
            if (errors) return false;
        } else if (expect_error != (errors == 1) || (expect_error && result) || (!expect_error && !result)) return false;
    }
    const auto stats = budget->snapshot();
    if (input.refs || live_outputs || stats.owned) return false;
    for (auto bytes : stats.bytes) if (bytes) return false;
    return true;
}
bool unequal_stride_adapter() {
    const auto api = make_api();
    VSFrame frame;
    frame.width = 16; frame.height = 3; frame.stride_bytes = 23 * sizeof(float);
    frame.pixels.resize(69, -123.f);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 16; ++x) frame.pixels[y * 23 + x] = y + x * .125f;
    const auto original = frame.pixels;
    auto budget = std::make_shared<nss::ResourceBudget>(1 << 20);
    {
        nss::ResourceScope resources(budget);
        nss::FrameScope frames(&api);
        if (frames.readPlane(&frame, 0, 23) != frame.pixels.data() || budget->snapshot().owned) return false;
        for (int pitch : {16, 19, 29}) {
            const float* mirror = frames.readPlane(&frame, 0, pitch);
            if (mirror == frame.pixels.data() || frames.readPlane(&frame, 0, pitch) != mirror) return false;
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < pitch; ++x)
                    if (mirror[y * pitch + x] != (x < 16 ? y + x * .125f : 0.f)) return false;
        }
        for (ptrdiff_t invalid : {ptrdiff_t(-4), ptrdiff_t(63), ptrdiff_t(60), ptrdiff_t(INT_MAX) * 4}) {
            frame.stride_bytes = invalid;
            bool rejected = false;
            try { frames.readPlane(&frame, 0, 16); } catch (const std::exception&) { rejected = true; }
            if (!rejected) return false;
        }
        frame.stride_bytes = 23 * sizeof(float);
    }
    if (frame.pixels != original || budget->snapshot().owned) return false;
    // The two allocations are the copied plane and the mirror index. Failure
    // at either position must release its reservation and permit a clean retry.
    for (int position : {1, 2}) {
        nss::ResourceScope resources(budget);
        {
            nss::FrameScope frames(&api);
            new_calls = 0; new_failure = position; count_new = true;
            bool rejected = false;
            try { frames.readPlane(&frame, 0, 29); } catch (const std::bad_alloc&) { rejected = true; }
            count_new = false;
            if (!rejected || budget->snapshot().owned) return false;
            if (frames.readPlane(&frame, 0, 29)[29 + 7] != 1.875f) return false;
        }
        if (budget->snapshot().owned) return false;
    }
    std::puts("production FrameScope: unequal strides, padding, reuse, rejection and allocation recovery passed");
    return true;
}
}
int workspace_test_memalign(void** ptr, std::size_t alignment, std::size_t bytes) {
    return fail_workspace ? ENOMEM : posix_memalign(ptr, alignment, bytes);
}
int main() {
    if (!unequal_stride_adapter()) return 1;
    input.pixels.resize(256, .25f);
    for (int failure = 1; failure <= 8; ++failure) {
        if (!run(failure, false, 1<<24, true)) { std::fprintf(stderr,"frame failure point %d leaked or escaped\n",failure); return 1; }
    }
    if (!run(0, true, 1<<24, true) || acquired != 7) { std::fprintf(stderr,"original seven-reference workspace ENOMEM failed\n"); return 1; }
    if (!run(0, false, 1, true) || !run(0, false, 1<<24, false)) return 1;
    int injected = 0;
    for (bool rolling : {false, true}) {
        if (!run(0, false, 1<<24, false, 0, rolling)) return 1;
        const int allocations = new_calls;
        for (int position = 1; position <= allocations; ++position) {
            if (!run(0, false, 1<<24, true, position, rolling)) {
                std::fprintf(stderr, "operator-new point %d rolling=%d leaked or escaped\n", position, rolling);
                return 1;
            }
            ++injected;
        }
    }
    std::printf("injected %d C++ allocation failures across fat and rolling callbacks\n", injected);
    std::puts("actual BM3D callback: get/new/workspace/budget failures released all references and allocations");
    return 0;
}
