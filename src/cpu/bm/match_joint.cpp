#include "nss/cpu_batch.hpp"
#include "cpu/batch_contract.hpp"
#include "cpu/hwy_config.hpp"
#include "cpu/bm/matcher.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "cpu/bm/match_joint.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace nss {
namespace HWY_NAMESPACE {
namespace hn=hwy::HWY_NAMESPACE;
// Each vector lane is a candidate, not a pixel. The accumulator buckets
// retain SsdBlock's pixel-lane recurrence, then mirror Highway 1.4 ReduceSum.
// This uses all lanes for 6/7/9 blocks without padding the semantic patch.
// Same-ISA exhaustive tests deliberately guard the pinned reduction contract.
template<class V>
static V ReduceSsdBuckets(const V* values) {
#if HWY_TARGET == HWY_EMU128
    const hn::ScalableTag<float> df;
    auto sum=hn::Zero(df);
    for(int i=0;i<static_cast<int>(hn::Lanes(df));++i) sum=hn::Add(sum,values[i]);
    return sum;
#elif HWY_TARGET == HWY_SCALAR
    return values[0];
#elif HWY_ARCH_ARM
    return hn::Add(hn::Add(values[0],values[1]),hn::Add(values[2],values[3]));
#elif HWY_MAX_BYTES >= 64
    const auto a=hn::Add(hn::Add(values[0],values[4]),hn::Add(values[12],values[8]));
    const auto b=hn::Add(hn::Add(values[1],values[5]),hn::Add(values[13],values[9]));
    const auto c=hn::Add(hn::Add(values[2],values[6]),hn::Add(values[14],values[10]));
    const auto d=hn::Add(hn::Add(values[3],values[7]),hn::Add(values[15],values[11]));
    return hn::Add(hn::Add(a,d),hn::Add(b,c));
#elif HWY_MAX_BYTES >= 32
    return hn::Add(hn::Add(hn::Add(values[0],values[4]),hn::Add(values[3],values[7])),
                   hn::Add(hn::Add(values[1],values[5]),hn::Add(values[2],values[6])));
#elif HWY_MAX_BYTES >= 16
    return hn::Add(hn::Add(values[0],values[3]),hn::Add(values[1],values[2]));
#else
    return values[0];
#endif
}

template<int B>
static void SpatialJoint4(const float* ref,int stride,int width,int height,
                          const MatchBatchItem* items,int count,Match* matches,int match_stride,int* counts) {
    const hn::ScalableTag<float> df;
    constexpr int lanes=HWY_LANES(float);
    using V=hn::Vec<decltype(df)>;
    float reference[4][B][B];
    int cx[4],cy[4],left[4],top[4],span_x[4],span_y[4],wanted[4];
    // Same total ordering as StableTopK, but an already-full group rejects
    // against its last element instead of rescanning every retained match.
    // This existing common primitive is already used by the ARM matcher.
    using TopK=detail::SpatialSortedTopK;
    std::optional<TopK> topk[4];
    int max_span_x=0,max_span_y=0;
    for(int q=0;q<count;++q) {
        cx[q]=std::clamp(items[q].bx,0,width-B);cy[q]=std::clamp(items[q].by,0,height-B);
        const int range=std::max(0,items[q].bm_range);
        left[q]=cx[q]-std::min(cx[q],range);top[q]=cy[q]-std::min(cy[q],range);
        span_x[q]=cx[q]+std::min(width-B-cx[q],range)-left[q]+1;
        span_y[q]=cy[q]+std::min(height-B-cy[q],range)-top[q]+1;
        wanted[q]=std::min(items[q].group,kBmMaxGroup);
        matches[q*match_stride]={cx[q],cy[q],0,0.f,0};
        topk[q].emplace(matches+q*match_stride+1,wanted[q]-1);
        if(wanted[q]>1) {
            max_span_x=std::max(max_span_x,span_x[q]);max_span_y=std::max(max_span_y,span_y[q]);
            for(int y=0;y<B;++y)for(int x=0;x<B;++x)
                reference[q][y][x]=ref[static_cast<std::size_t>(cy[q]+y)*stride+cx[q]+x];
        }
    }
    HWY_ALIGN float distances[lanes];
    for(int dy=0;dy<max_span_y;++dy)for(int dx=0;dx<max_span_x;dx+=lanes) {
        for(int q=0;q<count;++q) {
            if(wanted[q]<=1 || dy>=span_y[q] || dx>=span_x[q]) continue;
            const int valid=std::min(lanes,span_x[q]-dx);
            V buckets[lanes];
            for(int i=0;i<lanes;++i)buckets[i]=hn::Zero(df);
            const float* candidate=ref+static_cast<std::size_t>(top[q]+dy)*stride+left[q]+dx;
            for(int y=0;y<B;++y)for(int x=0;x<B;++x) {
                const auto value=hn::LoadN(df,candidate+y*stride+x,valid);
                const auto diff=hn::Sub(hn::Set(df,reference[q][y][x]),value);
                buckets[x%lanes]=hn::MulAdd(diff,diff,buckets[x%lanes]);
            }
            hn::StoreU(ReduceSsdBuckets(buckets),df,distances);
            for(int i=0;i<valid;++i) {
                const int x=left[q]+dx+i,y=top[q]+dy;
                if(x==cx[q] && y==cy[q])continue;
                const auto ordinal=static_cast<std::uint32_t>(1ull+static_cast<std::uint64_t>(dy)*span_x[q]+dx+i);
                topk[q]->add({x,y,0,distances[i],ordinal});
            }
        }
    }
    for(int q=0;q<count;++q) counts[q]=1+topk[q]->finish();
}

int SpatialMatchJoint(const float* ref,int stride,int width,int height,const MatchBatchItem* items,int count,
                      Match* matches,int match_stride,int* counts) {
    if(count==0) return 0;
    if(!ref || !items || !matches || !counts || count<0 || width<1 || height<1 || stride<width || match_stride<1) return -1;
    int first_error=0;
    for(int begin=0;begin<count;) {
        const int block=items[begin].block;
        int end=begin;
        // EMU128's emulated MulAdd inherits the original fast-math TU's
        // contraction policy. Keep its same-model reference recurrence;
        // native NEON/AVX2/AVX3 have an explicit fused instruction contract.
        constexpr bool native=HWY_TARGET!=HWY_EMU128 && HWY_TARGET!=HWY_SCALAR;
        if(native && (block==6 || block==7 || block==9)) {
            while(end<count && end<begin+4 && items[end].block==block && width>=block && height>=block &&
                  items[end].group>0 && std::min(items[end].group,kBmMaxGroup)<=match_stride) ++end;
        }
        if(end>begin) {
            auto* out=matches+static_cast<std::size_t>(begin)*match_stride;
            if(block==6) SpatialJoint4<6>(ref,stride,width,height,items+begin,end-begin,out,match_stride,counts+begin);
            if(block==7) SpatialJoint4<7>(ref,stride,width,height,items+begin,end-begin,out,match_stride,counts+begin);
            if(block==9) SpatialJoint4<9>(ref,stride,width,height,items+begin,end-begin,out,match_stride,counts+begin);
            begin=end;
        } else {
            const auto& item=items[begin];
            counts[begin]=!detail::match_capacity_valid(item.group, match_stride)?0:
                spatial_match(ref,stride,width,height,item.bx,item.by,item.block,item.bm_range,item.group,
                             matches+static_cast<std::size_t>(begin)*match_stride,item.avx2_features|NSS_AVX2_REQUESTED);
            if(counts[begin]<=0) detail::record_batch_failure(begin, first_error);
            ++begin;
        }
    }
    return first_error;
}

}  // namespace HWY_NAMESPACE
}  // namespace nss
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace nss {
HWY_EXPORT(SpatialMatchJoint);
int spatial_match_joint(const float* ref,int stride,int width,int height,const MatchBatchItem* items,int count,
                        Match* matches,int match_stride,int* counts) {
    return HWY_DYNAMIC_DISPATCH(SpatialMatchJoint)(ref,stride,width,height,items,count,matches,match_stride,counts);
}
}
#endif
