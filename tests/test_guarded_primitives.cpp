#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_linalg.hpp"
#include "nss/cpu_nlh.hpp"
#include "cpu/wnnm/jacobi8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace {
constexpr float sentinel = -9876.f;
constexpr int widths[] = {1,2,3,4,5,7,8,9,12,15,16,17,31,32,33};

// The last active element touches a PROT_NONE page: no accessible row padding
// can hide an overread/overwrite. A prefix canary catches short underruns.
struct Guarded {
    void* mapping;
    std::size_t bytes;
    int count;
    float* data;
    explicit Guarded(int n) : count(n) {
        const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        const auto pages = (static_cast<std::size_t>(n + 16) * sizeof(float) + page - 1) / page;
        bytes = (pages + 1) * page;
        mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap");
        auto* end = static_cast<char*>(mapping) + pages * page;
        if (mprotect(end, page, PROT_NONE)) { munmap(mapping, bytes); throw std::runtime_error("mprotect"); }
        data = reinterpret_cast<float*>(end) - n;
        std::fill(data - 16, data + n, sentinel);
    }
    ~Guarded() { munmap(mapping, bytes); }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    float& operator[](int i) { return data[i]; }
    void prefix() const {
        for (int i = -16; i < 0; ++i) if (data[i] != sentinel) throw std::runtime_error("prefix overwritten");
    }
};

void near(double got, double expected, double tolerance, const char* name) {
    if (!std::isfinite(got) || std::abs(got - expected) > tolerance) {
        std::fprintf(stderr, "%s: %.12g != %.12g (tol %.3g)\n", name, got, expected, tolerance);
        throw std::runtime_error(name);
    }
}
float sample(int i) { return static_cast<float>((i * 7) % 31 - 15) / 32.f; }
int extent(int width, int height, int stride) { return (height - 1) * stride + width; }
void padding(const Guarded& a, int width, int height, int stride) {
    a.prefix();
    for (int y = 0; y + 1 < height; ++y) for (int x = width; x < stride; ++x)
        if (a.data[y * stride + x] != sentinel) throw std::runtime_error("row padding overwritten");
}

void vectors() {
    Guarded empty(0);
    near(nss::dot_n(empty.data, empty.data, 0), 0, 0, "empty dot");
    near(nss::ssd_vec(empty.data, empty.data, 0), 0, 0, "empty SSD");
    nss::axpy_n(empty.data, empty.data, 1, 0);
    nss::scale_n(empty.data, 2, 0);
    nss::soft_threshold(empty.data, 0, 1);
    nss::soft_threshold_var(empty.data, empty.data, 0);
    nss::iter_regularize(empty.data, empty.data, 0, .5f);
    for (int n : widths) {
        Guarded a(n), b(n), y(n), tau(n);
        double dot = 0, ssd = 0;
        for (int i = 0; i < n; ++i) {
            a[i] = sample(i); b[i] = sample(i + 3); y[i] = b[i]; tau[i] = (i % 4 - 1) / 16.f;
            dot += double(a[i]) * b[i]; ssd += std::pow(double(a[i]) - b[i], 2);
        }
        near(nss::dot_n(a.data,b.data,n),dot,0,"dyadic dot tail");
        near(nss::ssd_vec(a.data,b.data,n),ssd,0,"dyadic SSD tail");
        nss::axpy_n(y.data,a.data,.5f,n);
        for(int i=0;i<n;++i) near(y[i],b[i]+.5*a[i],0,"axpy tail");
        nss::scale_n(y.data,.25f,n);
        for(int i=0;i<n;++i) near(y[i],.25*(b[i]+.5*a[i]),0,"scale tail");
        std::copy_n(a.data,n,y.data); nss::soft_threshold(y.data,n,.125f);
        for(int i=0;i<n;++i) near(y[i],std::copysign(std::max(std::abs(a[i])-.125f,0.f),a[i]),0,"threshold tail");
        std::copy_n(a.data,n,y.data); nss::soft_threshold_var(y.data,tau.data,n);
        for(int i=0;i<n;++i) near(y[i],std::copysign(std::max(std::abs(a[i])-std::max(tau[i],0.f),0.f),a[i]),0,"variable threshold tail");
        std::copy_n(a.data,n,y.data); nss::iter_regularize(y.data,b.data,n,.5f);
        for(int i=0;i<n;++i) near(y[i],a[i]+.5*(b[i]-a[i]),0,"regularization tail");
        a.prefix();b.prefix();y.prefix();tau.prefix();
    }
}

void groups_and_gemm() {
    for(int m:widths) for(int n:{1,3,4,5,8}) {
        const int lda=m+3;
        Guarded a(extent(m,n,lda)),mean(m);
        for(int j=0;j<n;++j) for(int i=0;i<m;++i) a[j*lda+i]=sample(i+j);
        nss::group_center_sub(a.data,m,n,lda,mean.data);
        for(int i=0;i<m;++i) {
            double mu=0;for(int j=0;j<n;++j)mu+=sample(i+j);mu/=n;
            near(mean[i],mu,2e-6,"group mean tail");
            for(int j=0;j<n;++j)near(a[j*lda+i],sample(i+j)-mu,2e-6,"group subtract tail");
        }
        nss::group_center_add(a.data,m,n,lda,mean.data);
        for(int j=0;j<n;++j)for(int i=0;i<m;++i)near(a[j*lda+i],sample(i+j),2e-6,"group restore tail");
        padding(a,m,n,lda);mean.prefix();
    }
    for(int m:widths) for(int n:{1,3,4,8,9}) for(int k:{0,1,3,8}) {
        const int lda=m+3,ldb=k+2,ldc=m+5;
        Guarded a(k?extent(m,k,lda):0),b(k?extent(k,n,ldb):0),c(extent(m,n,ldc));
        for(int t=0;t<k;++t)for(int i=0;i<m;++i)a[t*lda+i]=sample(i+t);
        for(int j=0;j<n;++j)for(int t=0;t<k;++t)b[j*ldb+t]=sample(j+t+4);
        nss::gemm_nn_hwy(m,n,k,a.data,lda,b.data,ldb,c.data,ldc);
        for(int j=0;j<n;++j)for(int i=0;i<m;++i){double value=0;for(int t=0;t<k;++t)value+=double(a[t*lda+i])*b[j*ldb+t];near(c[j*ldc+i],value,0,"GEMM tails/zero rank");}
        a.prefix();b.prefix();padding(c,m,n,ldc);
    }
}

void patches_and_ssd() {
    for(int block:{1,2,4,8,12,16,32}) {
        const int w=block+3,h=block+2,ss=w+3,ds=w+5;
        Guarded src(extent(w,h,ss)),num(extent(w,h,ds)),den(extent(w,h,ds)),patch(block*block);
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)src[y*ss+x]=sample(y*w+x);
        for(int y:{-2,0,h-block,h-1})for(int x:{-2,0,w-block,w-1}) {
            nss::pack_patch(patch.data,block*block,src.data,ss,x,y,block,w,h);
            for(int py=0;py<block;++py)for(int px=0;px<block;++px)
                near(patch[py*block+px],src[std::clamp(y+py,0,h-1)*ss+std::clamp(x+px,0,w-1)],0,"pack boundary");
            for(bool fixed:{false,true}) {
                for(int yy=0;yy<h;++yy)for(int xx=0;xx<w;++xx){num[yy*ds+xx]=.125f;den[yy*ds+xx]=.25f;}
                if(fixed) nss::unpack_patch_fixed(num.data,den.data,ds,x,y,patch.data,block,w,h,.5f,0);
                else nss::unpack_patch(num.data,den.data,ds,x,y,patch.data,block,w,h,.5f);
                for(int yy=0;yy<h;++yy)for(int xx=0;xx<w;++xx){bool hit=xx>=x&&xx<x+block&&yy>=y&&yy<y+block;near(num[yy*ds+xx],.125+(hit?.5*patch[(yy-y)*block+xx-x]:0),0,"unpack boundary");near(den[yy*ds+xx],hit?.75:.25,0,"unpack weight");}
            }
            padding(num,w,h,ds);padding(den,w,h,ds);patch.prefix();src.prefix();
        }
        std::vector<std::unique_ptr<Guarded>> a,b;
        const float* ap[3];const float* bp[3];int sa[3],sb[3];double expected=0;
        for(int ch=0;ch<3;++ch){sa[ch]=block+ch+1;sb[ch]=block+ch+5;a.emplace_back(new Guarded(extent(block,block,sa[ch])));b.emplace_back(new Guarded(extent(block,block,sb[ch])));ap[ch]=a.back()->data;bp[ch]=b.back()->data;
            for(int y=0;y<block;++y)for(int x=0;x<block;++x){(*a.back())[y*sa[ch]+x]=sample(x+y+ch);(*b.back())[y*sb[ch]+x]=sample(x+y+ch+3);expected+=std::pow(double(ap[ch][y*sa[ch]+x])-bp[ch][y*sb[ch]+x],2);}
            double single=0;for(int y=0;y<block;++y)for(int x=0;x<block;++x)single+=std::pow(double(ap[ch][y*sa[ch]+x])-bp[ch][y*sb[ch]+x],2);
            near(nss::ssd_block(ap[ch],sa[ch],bp[ch],sb[ch],block),single,0,"SSD independent strides");
        }
        near(nss::ssd_nch(ap,sa,bp,sb,3,block),expected,0,"three-channel SSD guards");
    }
}

void transforms() {
    for(int n:{1,2,4,8,12,16,32,64})for(int count:widths) {
        const int stride=count+1;
        Guarded a(extent(count,n,stride));
        for(int i=0;i<n;++i)for(int line=0;line<count;++line)a[i*stride+line]=sample(i+line);
        nss::dct_lines(a.data,n,1,stride,count,false);
        for(int k=0;k<n;++k)for(int line=0;line<count;++line){double value=0;for(int x=0;x<n;++x)value+=sample(x+line)*std::sqrt((k?2.:1.)/n)*std::cos(std::numbers::pi*(x+.5)*k/n);near(a[k*stride+line],value,2e-5,"DCT strided tail oracle");}
        padding(a,count,n,stride);
        nss::dct_lines(a.data,n,1,stride,count,true);
        for(int i=0;i<n;++i)for(int line=0;line<count;++line)near(a[i*stride+line],sample(i+line),2e-4,"DCT roundtrip tail");
        padding(a,count,n,stride);
        // The NLH Haar API accepts only 1/2/4/8/16, unlike BM3D's DCT axis.
        if(n<=16 && (n&(n-1))==0){Guarded x(n),y(n),z(n);for(int i=0;i<n;++i)x[i]=sample(i);nss::haar1d(x.data,y.data,n);nss::ihaar1d(y.data,z.data,n);for(int i=0;i<n;++i)near(z[i],x[i],2e-6,"Haar tail roundtrip");x.prefix();y.prefix();z.prefix();}
    }
}

void multichannel_patches() {
    for (int block : {1, 2, 4, 8, 12, 16, 32}) {
        const int w = block + 3, h = block + 2;
        Guarded patch(3 * block * block);
        std::vector<std::unique_ptr<Guarded>> src, num, den;
        const float* sources[3];
        float* nums[3];
        float* dens[3];
        int ss[3], ds[3];
        for (int c = 0; c < 3; ++c) {
            ss[c] = w + c + 1;
            ds[c] = w + c + 5;
            src.emplace_back(new Guarded(extent(w, h, ss[c])));
            num.emplace_back(new Guarded(extent(w, h, ds[c])));
            den.emplace_back(new Guarded(extent(w, h, ds[c])));
            sources[c] = src.back()->data;
            nums[c] = num.back()->data;
            dens[c] = den.back()->data;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    (*src[c])[y * ss[c] + x] = sample(c * 13 + y * w + x);
        }
        for (int y : {-2, 0, h - block, h - 1}) {
            for (int x : {-2, 0, w - block, w - 1}) {
                nss::pack_patch_nch(patch.data, patch.count, sources, ss, 3, x, y, block, w, h);
                for (int c = 0; c < 3; ++c) {
                    for (int py = 0; py < block; ++py) {
                        for (int px = 0; px < block; ++px) {
                            const int index = c * block * block + py * block + px;
                            const int sy = std::clamp(y + py, 0, h - 1);
                            const int sx = std::clamp(x + px, 0, w - 1);
                            near(patch[index], sources[c][sy * ss[c] + sx], 0, "channel-major pack");
                        }
                    }
                    for (int yy = 0; yy < h; ++yy)
                        for (int xx = 0; xx < w; ++xx) {
                            nums[c][yy * ds[c] + xx] = .125f;
                            dens[c][yy * ds[c] + xx] = .25f;
                        }
                }
                nss::unpack_patch_nch(nums, dens, ds, 3, x, y, patch.data, block, w, h, .5f);
                for (int c = 0; c < 3; ++c) {
                    for (int yy = 0; yy < h; ++yy) {
                        for (int xx = 0; xx < w; ++xx) {
                            const bool hit = xx >= x && xx < x + block && yy >= y && yy < y + block;
                            const int index = c * block * block + (yy - y) * block + xx - x;
                            near(nums[c][yy * ds[c] + xx], .125 + (hit ? .5 * patch[index] : 0), 0,
                                 "multichannel scatter");
                            near(dens[c][yy * ds[c] + xx], hit ? .75 : .25, 0, "multichannel weight");
                        }
                    }
                    padding(*src[c], w, h, ss[c]);
                    padding(*num[c], w, h, ds[c]);
                    padding(*den[c], w, h, ds[c]);
                }
                patch.prefix();
            }
        }
    }
}

void hq_primitives() {
    for(int n:widths) {
        constexpr int m=7,k=9;
        const int lda=k+3,ldb=n+2,ldc=n+4;
        Guarded a(extent(k,m,lda)),b(extent(n,k,ldb)),c(extent(n,m,ldc));
        for(int i=0;i<m;++i)for(int j=0;j<k;++j)a[i*lda+j]=sample(i+j);
        for(int i=0;i<k;++i)for(int j=0;j<n;++j)b[i*ldb+j]=sample(i*3+j);
        if(nss::gemm_f32_f64(a.data,lda,b.data,ldb,c.data,ldc,m,n,k)) throw std::runtime_error("precise GEMM view");
        for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
            double expected=0;for(int t=0;t<k;++t)expected+=a[i*lda+t]*b[t*ldb+j];
            near(c[i*ldc+j],static_cast<float>(expected),0,"precise GEMM guard");
        }
        padding(a,k,m,lda);padding(b,n,k,ldb);padding(c,n,m,ldc);
    }
    for(int block:{6,7,9})for(int extra:{0,1,3,7}) {
        const int w=block+extra,h=block+extra,st=w+5;
        Guarded input(extent(w,h,st));
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)input[y*st+x]=sample(y*w+x);
        nss::MatchBatchItem jobs[5];nss::Match output[5*32];int counts[5];
        for(int i=0;i<5;++i)jobs[i]={extra*i/4,extra*i/4,block,30,32,0};
        if(nss::spatial_match_joint(input.data,st,w,h,jobs,5,output,32,counts)) throw std::runtime_error("joint match guard");
        padding(input,w,h,st);
    }
}

void aggregation() {
    for(int w:widths) {
        constexpr int h=3;const int ss=w+1,bs=w+3,ds=w+5;
        Guarded src(extent(w,h,ss)),num(extent(w,h,bs)),den(extent(w,h,bs)),dst(extent(w,h,ds));
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){src[y*ss+x]=sample(y+x);den[y*bs+x]=x%3;num[y*bs+x]=(.125f*y+x/64.f)*den[y*bs+x];}
        nss::aggregate_finish(dst.data,num.data,den.data,src.data,w,h,ds,bs,ss);
        for(int y=0;y<h;++y)for(int x=0;x<w;++x)near(dst[y*ds+x],x%3?.125*y+x/64.:src[y*ss+x],2e-6,"aggregate guard/zero den");
        for(int count:{1,3,33}) {
            std::vector<std::unique_ptr<Guarded>> ns,de;std::vector<const float*> np,dp;std::vector<int> strides;
            for(int c=0;c<count;++c){const int st=w+c+1;strides.push_back(st);ns.emplace_back(new Guarded(extent(w,h,st)));de.emplace_back(new Guarded(extent(w,h,st)));np.push_back(ns.back()->data);dp.push_back(de.back()->data);
                for(int y=0;y<h;++y)for(int x=0;x<w;++x){(*de.back())[y*st+x]=x%3?c%3+1:0;(*ns.back())[y*st+x]=(.125f*y+x/64.f)*(*de.back())[y*st+x];}
            }
            nss::vaggregate_target(dst.data,np.data(),dp.data(),strides.data(),count,src.data,w,h,ds,ss);
            for(int y=0;y<h;++y)for(int x=0;x<w;++x)near(dst[y*ds+x],x%3?.125*y+x/64.:src[y*ss+x],2e-6,"destination aggregate guards");
        }
        src.prefix();num.prefix();den.prefix();padding(dst,w,h,ds);
    }
}
} // namespace

int main() {
    try { vectors(); groups_and_gemm(); patches_and_ssd(); multichannel_patches(); transforms(); aggregation(); hq_primitives(); }
    catch(const std::exception& e){std::fprintf(stderr,"guarded primitives: %s\n",e.what());return 1;}
    std::puts("guard-page tails, independent strides, dyadic arithmetic, DCT oracle, pack/scatter and aggregation passed");
}
