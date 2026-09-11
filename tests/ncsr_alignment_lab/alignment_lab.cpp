#include "nss/ncsr_alignment_lab.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/avx2_policy.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace nss::alignment_lab {
namespace {
using Clock = std::chrono::steady_clock;
using Vec = std::vector<float>;
struct Context {
    int flags = 0;
    float input_sigma = 0, match_sigma = 0;
    double tau_sum = 0;
    std::uint64_t tau_count = 0;
};
thread_local Context* context = nullptr;
struct ContextScope {
    Context* previous;
    explicit ContextScope(Context& current) : previous(context) { context = &current; }
    ~ContextScope() { context = previous; }
};
double elapsed(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
float coefficient(float sigma) { return sigma*255 <= 10 ? .56f : sigma*255 <= 15 ? .59f : .64f; }
float noise_factor(float sigma) { return sigma*255 <= 15 ? .22f : sigma*255 <= 30 ? .23f : .26f; }
float residual_sigma(float initial, const Vec& estimate, const Vec& noisy) {
    double energy=0;
    for (std::size_t i=0; i<estimate.size(); ++i) {
        double error=static_cast<double>(estimate[i])-noisy[i]; energy += error*error;
    }
    // Conservative engineering clamp; the author program uses abs here.
    return noise_factor(initial)*std::sqrt(std::max(0., static_cast<double>(initial)*initial-energy/estimate.size()));
}
void weights_for(const float* distances, int n, int m, float sigma, float* weights) {
    const bool aligned = context && (context->flags & Weights);
    const float match_sigma = aligned ? context->match_sigma : context->input_sigma;
    const float sigma8 = match_sigma*255;
    const float hp = context->input_sigma*255 <= 15 ? 75.f : context->input_sigma*255 <= 30 ? 80.f :
                     context->input_sigma*255 <= 50 ? 90.f : 95.f;
    const float h = aligned ? m*std::max(12.f*sigma8,hp)/(255.f*255.f) :
                             std::max(2.f*m*match_sigma*match_sigma,1e-12f);
    int minimum=0;
    for(int j=1;j<n;++j) if(distances[j]<distances[minimum]) minimum=j;
    float second=std::numeric_limits<float>::infinity();
    for(int j=0;j<n;++j) if(j!=minimum) second=std::min(second,distances[j]);
    double sum=0;
    for(int j=0;j<n;++j) {
        const float distance = aligned && j==minimum && n>1 ? second : distances[j];
        weights[j]=std::exp(-distance/h); sum+=weights[j];
    }
    if (!(sum>0)) { for(int j=0;j<n;++j) weights[j]=1.f/n; }
    else for(int j=0;j<n;++j) weights[j]=static_cast<float>(weights[j]/sum);
}
void shrink_codes(float* codes,int r,int n,float sigma,const float* distances,int m) {
    std::vector<float> weights(n), beta(r), thresholds(r);
    weights_for(distances,n,m,sigma,weights.data());
    const bool aligned = context->flags & Statistics;
    const double sigma2=static_cast<double>(sigma)*sigma;
    for(int d=0;d<r;++d) {
        double mean=0;
        for(int j=0;j<n;++j) mean += weights[j]*codes[j*r+d];
        beta[d]=static_cast<float>(mean);
        double variance=0;
        for(int j=0;j<n;++j) {
            const double error=codes[j*r+d]-mean;
            variance += error*error*(aligned ? 1./n : weights[j]);
        }
        if(aligned) variance=std::max(0.,variance-sigma2);
        const double c=aligned ? coefficient(context->input_sigma)*std::sqrt(2.) : 2*std::sqrt(2.);
        const double tau=c*sigma2/(std::sqrt(variance)+1e-12);
        thresholds[d]=static_cast<float>(tau);
        context->tau_sum += tau; ++context->tau_count;
    }
    for(int j=0;j<n;++j) for(int d=0;d<r;++d) {
        const float residual=codes[j*r+d]-beta[d];
        codes[j*r+d]=beta[d]+std::copysign(std::max(0.f,std::fabs(residual)-thresholds[d]),residual);
    }
}

// Standard symmetric cyclic Jacobi eigensolver, independently implemented.
// Columns of eigenvectors are returned in descending eigenvalue order as rows.
Vec eigenbasis(std::vector<double> matrix,int d) {
    std::vector<double> vectors(d*d,0.);
    for(int i=0;i<d;++i) vectors[i*d+i]=1.;
    bool converged=false;
    for(int sweep=0;sweep<64;++sweep) {
        double off=0,scale=0;
        for(int i=0;i<d;++i) {
            scale=std::max(scale,std::abs(matrix[i*d+i]));
            for(int j=i+1;j<d;++j) off=std::max(off,std::abs(matrix[i*d+j]));
        }
        if(off<=1e-11*std::max(scale,1e-15)) { converged=true; break; }
        const double threshold=sweep<4 ? off*.02 : 0.;
        for(int p=0;p<d-1;++p) for(int q=p+1;q<d;++q) {
            const double apq=matrix[p*d+q];
            if(std::abs(apq)<=std::max(threshold,1e-16*std::max(scale,1e-15))) continue;
            const double app=matrix[p*d+p],aqq=matrix[q*d+q];
            const double theta=(aqq-app)/(2*apq);
            const double tangent=std::copysign(1.,theta)/(std::abs(theta)+std::hypot(1.,theta));
            const double c=1/std::hypot(1.,tangent),s=tangent*c;
            for(int k=0;k<d;++k) if(k!=p && k!=q) {
                const double akp=matrix[k*d+p],akq=matrix[k*d+q];
                matrix[k*d+p]=matrix[p*d+k]=c*akp-s*akq;
                matrix[k*d+q]=matrix[q*d+k]=s*akp+c*akq;
            }
            matrix[p*d+p]=app-tangent*apq; matrix[q*d+q]=aqq+tangent*apq;
            matrix[p*d+q]=matrix[q*d+p]=0;
            for(int k=0;k<d;++k) {
                const double vp=vectors[k*d+p],vq=vectors[k*d+q];
                vectors[k*d+p]=c*vp-s*vq; vectors[k*d+q]=s*vp+c*vq;
            }
        }
    }
    if(!converged) throw std::runtime_error("NCSR lab: PCA eigensolver did not converge");
    std::vector<int> order(d); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return matrix[a*d+a]>matrix[b*d+b];});
    Vec result(d*d);
    for(int row=0;row<d;++row) for(int col=0;col<d;++col) result[row*d+col]=static_cast<float>(vectors[col*d+order[row]]);
    return result;
}
Vec dct_basis(int block) {
    const int d=block*block; Vec result(d*d);
    const double pi=std::acos(-1.);
    for(int fy=0;fy<block;++fy) for(int fx=0;fx<block;++fx)
        for(int y=0;y<block;++y) for(int x=0;x<block;++x) {
            double scale=std::sqrt((fy?2.:1.)/block)*std::sqrt((fx?2.:1.)/block);
            result[(fy*block+fx)*d+y*block+x]=static_cast<float>(scale*std::cos(pi*(y+.5)*fy/block)*std::cos(pi*(x+.5)*fx/block));
        }
    return result;
}
struct Dictionaries { std::vector<Vec> bases; Vec centers; int clusters=0,training_samples=0; double flat_limit=0; };
float feature(const float* patch,int d,float* out) {
    double mean=0,energy=0;
    for(int i=0;i<d;++i) mean+=patch[i]; mean/=d;
    for(int i=0;i<d;++i) {out[i]=patch[i]-static_cast<float>(mean);energy+=out[i]*out[i];}
    return static_cast<float>(energy/d);
}
int classify(const float* f,int d,const Vec& centers,int k) {
    double best=std::numeric_limits<double>::infinity(); int winner=0;
    for(int c=0;c<k;++c) {
        double distance=0;
        for(int j=0;j<d;++j) {double e=f[j]-centers[c*d+j];distance+=e*e;}
        if(distance<best){best=distance;winner=c;}
    }
    return winner;
}
Dictionaries train(const Vec& image,int width,int height,int block,float sigma) {
    const int d=block*block,nx=width-block+1,ny=height-block+1;
    const int total=nx*ny,samples=std::min(total,8192);
    Vec raw(samples*d),features(samples*d),patch(d); std::vector<int> nonflat;
    Dictionaries result; result.training_samples=samples;
    result.flat_limit=static_cast<double>(sigma)*sigma+16./(255.*255.);
    for(int i=0;i<samples;++i) {
        const int index=static_cast<int>(static_cast<long long>(i)*total/samples);
        pack_patch(raw.data()+i*d,d,image.data(),width,index%nx,index/nx,block,width,height);
        if(feature(raw.data()+i*d,d,features.data()+i*d)>=result.flat_limit) nonflat.push_back(i);
    }
    result.clusters=std::min(70,static_cast<int>(nonflat.size()));
    const int k=result.clusters;
    result.centers.resize(k*d);
    for(int c=0;c<k;++c) {
        int index=nonflat[static_cast<std::size_t>(c)*nonflat.size()/k];
        std::copy_n(features.data()+index*d,d,result.centers.data()+c*d);
    }
    std::vector<int> labels(samples,0),counts(k+1);
    // Bounded single-scale, six-Lloyd-step adaptation, not the author's training recipe.
    for(int iteration=0;iteration<6 && k>0;++iteration) {
        std::vector<double> sum(k*d,0.); std::fill(counts.begin(),counts.end(),0);
        for(int index:nonflat) {
            const int cls=classify(features.data()+index*d,d,result.centers,k);
            labels[index]=cls+1; ++counts[cls+1];
            for(int j=0;j<d;++j) sum[cls*d+j]+=features[index*d+j];
        }
        for(int c=0;c<k;++c) if(counts[c+1]) for(int j=0;j<d;++j)
            result.centers[c*d+j]=static_cast<float>(sum[c*d+j]/counts[c+1]);
    }
    for(int index:nonflat) labels[index]=1+classify(features.data()+index*d,d,result.centers,k);
    result.bases.resize(k+1);
    for(int cls=0;cls<=k;++cls) {
        std::vector<int> members;
        for(int i=0;i<samples;++i) if(labels[i]==cls) members.push_back(i);
        if(members.size()<static_cast<std::size_t>(d+1)) {result.bases[cls]=dct_basis(block);continue;}
        std::vector<double> mean(d,0.),cov(d*d,0.);
        for(int i:members) for(int j=0;j<d;++j) mean[j]+=raw[i*d+j];
        for(double& value:mean) value/=members.size();
        for(int i:members) for(int a=0;a<d;++a) {
            const double va=raw[i*d+a]-mean[a];
            for(int b=0;b<=a;++b) cov[a*d+b]+=va*(raw[i*d+b]-mean[b]);
        }
        for(int a=0;a<d;++a) for(int b=0;b<=a;++b) cov[a*d+b]=cov[b*d+a]=cov[a*d+b]/(members.size()-1);
        // Isotropic noise subtraction does not change full orthogonal eigenvectors.
        for(int j=0;j<d;++j) cov[j*d+j]-=static_cast<double>(sigma)*sigma;
        result.bases[cls]=eigenbasis(std::move(cov),d);
    }
    return result;
}
struct Neighborhood {
    int x=0,y=0,count=0,target=0,cls=0;
    std::array<Match,kWnnmMaxGroup> matches{};
};
std::vector<Neighborhood> match_all(const Vec& image,int width,int height,const SearchConfig& cfg) {
    std::vector<Neighborhood> jobs;
    for(int y=0;y<height-cfg.block+cfg.step;y+=cfg.step) for(int x=0;x<width-cfg.block+cfg.step;x+=cfg.step) {
        Neighborhood job;job.x=std::min(x,width-cfg.block);job.y=std::min(y,height-cfg.block); jobs.push_back(job);
    }
    for(std::size_t start=0;start<jobs.size();start+=32) {
        const int count=static_cast<int>(std::min<std::size_t>(32,jobs.size()-start));
        std::array<MatchBatchItem,32> items{}; std::array<int,32> counts{};
        std::array<Match,32*kWnnmMaxGroup> matches{};
        for(int i=0;i<count;++i) items[i]={jobs[start+i].x,jobs[start+i].y,cfg.block,cfg.bm_range,cfg.group,
            detail::avx2_policy(detail::Avx2Algorithm::NCSR,cfg.block,cfg.group,0,false,0)};
        if(spatial_match_batch(image.data(),width,width,height,items.data(),count,matches.data(),kWnnmMaxGroup,counts.data()))
            throw std::runtime_error("NCSR lab: matching failed");
        for(int i=0;i<count;++i) {
            auto& job=jobs[start+i]; job.count=counts[i];job.target=-1;
            if(job.count<1) throw std::runtime_error("NCSR lab: empty match");
            for(int j=0;j<job.count;++j) {
                job.matches[j]=matches[i*kWnnmMaxGroup+j];
                if(job.matches[j].x==job.x && job.matches[j].y==job.y) job.target=j;
            }
            if(job.target<0) {
                job.target=job.count-1;
                job.matches[job.target].x=job.x;job.matches[job.target].y=job.y;job.matches[job.target].t=0;job.matches[job.target].dist=0;
            }
        }
    }
    return jobs;
}
void assign(std::vector<Neighborhood>& jobs,const Dictionaries& dict,const Vec& image,int width,int height,int block) {
    const int d=block*block;Vec patch(d),f(d);
    for(auto& job:jobs) {
        pack_patch(patch.data(),d,image.data(),width,job.x,job.y,block,width,height);
        const float variance=feature(patch.data(),d,f.data());
        job.cls=variance<dict.flat_limit || dict.clusters==0 ? 0 : 1+classify(f.data(),d,dict.centers,dict.clusters);
    }
}
void run_targets(const Vec& image,Vec& num,Vec& den,int width,int height,const SearchConfig& cfg,
                 float sigma,const std::vector<Neighborhood>& jobs,const Dictionaries* dict) {
    const int d=cfg.block*cfg.block,lda=(d+15)&~15,g=cfg.group;
    const int work_size=ncsr_filter_work_floats(d,g);
    for(std::size_t begin=0;begin<jobs.size();begin+=32) {
        const int count=static_cast<int>(std::min<std::size_t>(32,jobs.size()-begin));
        Vec patches(static_cast<std::size_t>(count)*lda*g),distances(count*g),work(static_cast<std::size_t>(count)*work_size);
        std::array<NcsrFilterBatchItem,32> items{};std::array<int,32> status{};
        for(int i=0;i<count;++i) {
            const auto& job=jobs[begin+i]; float* p=patches.data()+static_cast<std::size_t>(i)*lda*g;
            for(int j=0;j<job.count;++j) {
                pack_patch(p+j*lda,lda,image.data(),width,job.matches[j].x,job.matches[j].y,cfg.block,width,height);
                distances[i*g+j]=job.matches[j].dist;
            }
            items[i]={p,d,job.count,lda,sigma,distances.data()+i*g,work.data()+static_cast<std::size_t>(i)*work_size,work_size,&status[i]};
        }
        if(!dict && ncsr_filter_group_batch(items.data(),count)) throw std::runtime_error("NCSR lab: group filtering failed");
        for(int i=0;i<count;++i) {
            const auto& job=jobs[begin+i];float* patch=patches.data()+static_cast<std::size_t>(i)*lda*g;
            if(dict) {
                const auto& basis=dict->bases[job.cls];Vec codes(d*job.count);
                for(int j=0;j<job.count;++j) for(int row=0;row<d;++row) {
                    double sum=0;for(int col=0;col<d;++col) sum+=basis[row*d+col]*patch[j*lda+col];
                    codes[j*d+row]=static_cast<float>(sum);
                }
                shrink_codes(codes.data(),d,job.count,sigma,distances.data()+i*g,d);
                for(int col=0;col<d;++col) {
                    double sum=0;for(int row=0;row<d;++row) sum+=basis[row*d+col]*codes[job.target*d+row];
                    patch[job.target*lda+col]=static_cast<float>(sum);
                }
            }
            aggregate_add(num.data(),den.data(),width,job.x,job.y,patch+job.target*lda,cfg.block,width,height,1.f);
        }
    }
}
}

bool finish_codes(float* codes,int dimensions,int columns,float sigma,const float* distances,int m) {
    if(!context || !(context->flags & (Statistics|Weights))) return false;
    if(!distances) throw std::runtime_error("NCSR lab: explicit matching distances required");
    shrink_codes(codes,dimensions,columns,sigma,distances,m);return true;
}

void denoise(const float* input,int width,int height,int stride,float* output,int output_stride,
             SearchConfig cfg,float sigma,int iterations,float delta,int flags,std::string& trace) {
    if(flags<1 || flags>31 || ((flags&Dictionary) && !(flags&TargetOnly)) || ((flags&Reuse) && !(flags&TargetOnly)))
        throw std::runtime_error("NCSR lab: unsupported flag combination");
    if(cfg.radius!=0 || cfg.block>16 || cfg.block<1 || cfg.step<1 || cfg.step>cfg.block || cfg.group>32 || cfg.group<1 ||
       width<cfg.block || height<cfg.block || iterations<1 || iterations>12)
        throw std::runtime_error("NCSR lab: spatial bounded shape required");
    const int pixels=width*height,d=cfg.block*cfg.block,lda=(d+15)&~15;
    Vec estimate(pixels),noisy(pixels),num(pixels),den(pixels),patches(lda*cfg.group),work(ncsr_filter_work_floats(d,cfg.group));
    for(int y=0;y<height;++y) std::copy_n(input+y*stride,width,noisy.data()+y*width);
    estimate=noisy;
    Context state;state.flags=flags;state.input_sigma=sigma;state.match_sigma=sigma;ContextScope scope(state);
    std::vector<Neighborhood> jobs;Dictionaries dictionaries;std::ostringstream log;
    log.precision(10);log<<"{\"flags\":"<<flags<<",\"iterations\":[";
    for(int iteration=0;iteration<iterations;++iteration) {
        double training_seconds=0,matching_seconds=0; const bool refresh=!(flags&Reuse) || iteration%3==0;
        const float previous_sigma=state.match_sigma;
        if((flags&TargetOnly) && refresh) {
            if(flags&Dictionary) {auto start=Clock::now();dictionaries=train(estimate,width,height,cfg.block,previous_sigma);training_seconds=elapsed(start);}
            auto start=Clock::now();jobs=match_all(estimate,width,height,cfg);
            if(flags&Dictionary) assign(jobs,dictionaries,estimate,width,height,cfg.block);
            matching_seconds=elapsed(start);
        }
        if(iteration>0) iter_regularize(estimate.data(),noisy.data(),pixels,delta);
        float effective_sigma=(flags&Statistics) && iteration>0 ? residual_sigma(sigma,estimate,noisy) : sigma;
        // Match bandwidth is tied to the estimate used to build this neighborhood,
        // not overwritten by this iteration's post-feedback residual estimate.
        std::fill(num.begin(),num.end(),0.f);std::fill(den.begin(),den.end(),0.f);
        const auto filtering_start=Clock::now();const double old_tau=state.tau_sum;const auto old_count=state.tau_count;
        if(flags&TargetOnly) run_targets(estimate,num,den,width,height,cfg,effective_sigma,jobs,(flags&Dictionary)?&dictionaries:nullptr);
        else {
            const float* refs[1]={estimate.data()};int strides[1]={width};
            ncsr_run_groups(refs,strides,refs,strides,1,0,width,height,cfg,effective_sigma,num.data(),den.data(),patches.data(),work.data());
        }
        aggregate_finish(estimate.data(),num.data(),den.data(),estimate.data(),width,height,width,width);
        for(float value:estimate) if(!std::isfinite(value)) throw std::runtime_error("NCSR lab: nonfinite output");
        const auto tau_count=state.tau_count-old_count;
        if(iteration)log<<",";
        log<<"{\"sigma8\":"<<effective_sigma*255<<",\"training_seconds\":"<<training_seconds<<",\"matching_seconds\":"<<matching_seconds
           <<",\"filtering_seconds\":"<<elapsed(filtering_start)<<",\"mean_tau\":"<<(tau_count?(state.tau_sum-old_tau)/tau_count:0)
           <<",\"dictionary_clusters\":"<<dictionaries.clusters<<",\"training_samples\":"<<dictionaries.training_samples<<"}";
        // Cached neighborhoods retain their construction-time bandwidth.
        if(!(flags&Reuse) || (iteration+1)%3==0) state.match_sigma=effective_sigma;
    }
    log<<"]}";trace=log.str();
    for(int y=0;y<height;++y) std::copy_n(estimate.data()+y*width,width,output+y*output_stride);
}

bool self_test() {
    const std::vector<double> covariance={4,1,.5,1,2,.3,.5,.3,1};
    Vec basis=eigenbasis(covariance,3);
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        double dot=0;for(int k=0;k<3;++k)dot+=basis[i*3+k]*basis[j*3+k];
        if(std::abs(dot-(i==j))>1e-5)return false;
        if(i!=j) {
            double cross=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)cross+=basis[i*3+a]*covariance[a*3+b]*basis[j*3+b];
            if(std::abs(cross)>1e-5)return false;
        }
    }
    Context test;test.flags=Statistics|Weights;test.input_sigma=.1f;test.match_sigma=.1f;ContextScope scope(test);
    float distances[3]={0,.1f,100},weights[3];weights_for(distances,3,16,.1f,weights);
    if(std::abs(weights[0]-weights[1])>1e-6 || weights[2]>.0001)return false;
    float codes[3]={.3f,.3f,.9f};shrink_codes(codes,1,3,0.f,distances,16);
    if(std::abs(codes[2]-.9f)>1e-6)return false;
    Vec noisy(16,.5f),estimate=noisy;
    if(std::abs(residual_sigma(.1f,estimate,noisy)-.1f*noise_factor(.1f))>1e-7)return false;
    for(int block : {6,7,8,9}) {
        const int d=block*block;
        std::vector<double> matrix(d*d,0.);
        for(int i=0;i<d;++i)for(int j=0;j<d;++j) {
            for(int k=0;k<d+3;++k) {
                const double a=std::sin((i+1)*(k+1)*.371),b=std::sin((j+1)*(k+1)*.371);
                matrix[i*d+j]+=a*b/(d+3);
            }
        }
        auto p=eigenbasis(matrix,d);
        for(int i=0;i<d;++i)for(int j=0;j<d;++j) {
            double dot=0;for(int k=0;k<d;++k)dot+=p[i*d+k]*p[j*d+k];
            if(std::abs(dot-(i==j))>2e-5)return false;
        }
        for(int i=0;i<d;++i) {
            std::vector<double> projected(d,0.);double eigenvalue=0;
            for(int a=0;a<d;++a)for(int b=0;b<d;++b)projected[a]+=matrix[a*d+b]*p[i*d+b];
            for(int a=0;a<d;++a)eigenvalue+=projected[a]*p[i*d+a];
            for(int a=0;a<d;++a)if(std::abs(projected[a]-eigenvalue*p[i*d+a])>2e-5)return false;
        }
    }
    {
        constexpr int width=24,height=24;Vec input(width*height),output(width*height);
        for(int i=0;i<width*height;++i) input[i]=.5f+.25f*std::sin(i*.71f);
        SearchConfig cfg;cfg.block=8;cfg.step=4;cfg.group=8;cfg.bm_range=3;cfg.radius=0;
        std::string log;
        denoise(input.data(),width,height,width,output.data(),width,cfg,0.f,2,.1f,31,log);
        for(int i=0;i<width*height;++i)if(std::abs(input[i]-output[i])>3e-5)return false;
    }
    return true;
}
}
