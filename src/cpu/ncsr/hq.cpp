#include "nss/ncsr_hq.hpp"
#include "nss/resources.hpp"
#include "nss/checked.hpp"
#include "nss/cpu_ncsr.hpp"
#include "nss/cpu_batch.hpp"
#include "nss/cpu_api.hpp"
#include "nss/cpu_common.hpp"
#include "nss/cpu_linalg.hpp"
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

namespace nss::ncsr_hq {
namespace {
using Clock = std::chrono::steady_clock;
using Vec = ResourceVector<float>;
struct Context {
    float input_sigma = 0, match_sigma = 0;
    double tau_sum = 0;
    std::uint64_t tau_count = 0;
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
void weights_for(const Context& state, const float* distances, int n, int m, float* weights) {
    const float match_sigma = state.match_sigma;
    const float sigma8 = match_sigma*255;
    const float hp = state.input_sigma*255 <= 15 ? 75.f : state.input_sigma*255 <= 30 ? 80.f :
                     state.input_sigma*255 <= 50 ? 90.f : 95.f;
    const float h = m*std::max(12.f*sigma8,hp)/(255.f*255.f);
    int minimum=0;
    for(int j=1;j<n;++j) if(distances[j]<distances[minimum]) minimum=j;
    float second=std::numeric_limits<float>::infinity();
    for(int j=0;j<n;++j) if(j!=minimum) second=std::min(second,distances[j]);
    double sum=0;
    for(int j=0;j<n;++j) {
        const float distance = j==minimum && n>1 ? second : distances[j];
        weights[j]=std::exp(-distance/h); sum+=weights[j];
    }
    if (!(sum>0)) { for(int j=0;j<n;++j) weights[j]=1.f/n; }
    else for(int j=0;j<n;++j) weights[j]=static_cast<float>(weights[j]/sum);
}
void shrink_target(Context& state, float* codes,int r,int n,int target,float sigma,const float* distances,
                   double* tau_values=nullptr) {
    std::array<float,32> weights{};
    std::array<float,256> beta{}, thresholds{};
    weights_for(state,distances,n,r,weights.data());
    const double sigma2=static_cast<double>(sigma)*sigma;
    for(int d=0;d<r;++d) {
        double mean=0;
        for(int j=0;j<n;++j) mean += weights[j]*codes[j*r+d];
        beta[d]=static_cast<float>(mean);
        double variance=0;
        for(int j=0;j<n;++j) {
            const double error=codes[j*r+d]-mean;
            variance += error*error*(1./n);
        }
        variance=std::max(0.,variance-sigma2);
        const double c=coefficient(state.input_sigma)*std::sqrt(2.);
        const double tau=c*sigma2/(std::sqrt(variance)+1e-12);
        thresholds[d]=static_cast<float>(tau);
        if(tau_values) tau_values[d]=tau;
        else {state.tau_sum += tau; ++state.tau_count;}
    }
    const int j=target;
    for(int d=0;d<r;++d) {
        const float residual=codes[j*r+d]-beta[d];
        codes[j*r+d]=beta[d]+std::copysign(std::max(0.f,std::fabs(residual)-thresholds[d]),residual);
    }
}

// Standard symmetric cyclic Jacobi eigensolver, independently implemented.
// Columns of eigenvectors are returned in descending eigenvalue order as rows.
Vec eigenbasis(ResourceVector<double> matrix,int d) {
    ResourceVector<double> vectors(d*d,0.);
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
    ResourceVector<int> order(d); std::iota(order.begin(),order.end(),0);
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
struct Dictionaries { ResourceVector<Vec> bases,transposes; Vec centers; int clusters=0,training_samples=0; double flat_limit=0; };
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
    Vec raw(samples*d),features(samples*d),patch(d); ResourceVector<int> nonflat;
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
    ResourceVector<int> labels(samples,0),counts(k+1);
    // Bounded single-scale, six-Lloyd-step adaptation, not the author's training recipe.
    for(int iteration=0;iteration<6 && k>0;++iteration) {
        ResourceVector<double> sum(k*d,0.); std::fill(counts.begin(),counts.end(),0);
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
        ResourceVector<int> members;
        for(int i=0;i<samples;++i) if(labels[i]==cls) members.push_back(i);
        if(members.size()<static_cast<std::size_t>(d+1)) continue;
        ResourceVector<double> mean(d,0.),cov(d*d,0.);
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
    result.transposes.resize(k+1);
    for(int cls=0;cls<=k;++cls)if(!result.bases[cls].empty()) {
        auto& transpose=result.transposes[cls];transpose.resize(d*d);
        for(int row=0;row<d;++row)for(int col=0;col<d;++col) transpose[row*d+col]=result.bases[cls][col*d+row];
    }
    return result;
}
struct Neighbor { int position; float distance; };
struct Neighborhood { int x, y, count=0, target=0, cls=0; std::size_t offset=0; };
struct MatchPlan {
    ResourceVector<Neighborhood> jobs;
    ResourceVector<Neighbor> neighbors;
};

void match_all(MatchPlan& plan,const Vec& image,int width,int height,const SearchConfig& cfg) {
    plan.jobs.clear(); plan.neighbors.clear();
    const std::size_t nx=(width-cfg.block+cfg.step-1)/cfg.step+1;
    const std::size_t ny=(height-cfg.block+cfg.step-1)/cfg.step+1;
    plan.jobs.reserve(checked_mul(nx,ny));
    // Reserve the actual maximum neighborhood, not the solver's fixed 32 slots.
    const auto search_side=2ull*static_cast<unsigned>(std::max(0,cfg.bm_range))+1;
    const auto max_neighbors=std::min<std::uint64_t>(cfg.group,
        std::min<std::uint64_t>(width-cfg.block+1,search_side)*
        std::min<std::uint64_t>(height-cfg.block+1,search_side));
    plan.neighbors.reserve(checked_mul(checked_mul(nx,ny),max_neighbors));
    for(int y=0;y<height-cfg.block+cfg.step;y+=cfg.step)
        for(int x=0;x<width-cfg.block+cfg.step;x+=cfg.step)
            plan.jobs.push_back({std::min(x,width-cfg.block),std::min(y,height-cfg.block)});
    for(std::size_t start=0;start<plan.jobs.size();start+=32) {
        const int count=static_cast<int>(std::min<std::size_t>(32,plan.jobs.size()-start));
        std::array<MatchBatchItem,32> items{}; std::array<int,32> counts{};
        std::array<Match,32*kWnnmMaxGroup> matches{};
        for(int i=0;i<count;++i) items[i]={plan.jobs[start+i].x,plan.jobs[start+i].y,cfg.block,cfg.bm_range,cfg.group,
            detail::avx2_policy(detail::Avx2Algorithm::NCSR,cfg.block,cfg.group,0,false,0)};
        if(spatial_match_joint(image.data(),width,width,height,items.data(),count,matches.data(),kWnnmMaxGroup,counts.data()))
            throw std::runtime_error("NCSR HQ: matching failed");
        for(int i=0;i<count;++i) {
            auto& job=plan.jobs[start+i]; job.count=counts[i];job.target=-1;job.offset=plan.neighbors.size();
            if(job.count<1 || job.count>cfg.group) throw std::runtime_error("NCSR HQ: invalid match count");
            for(int j=0;j<job.count;++j) {
                const auto& match=matches[i*kWnnmMaxGroup+j];
                plan.neighbors.push_back({match.y*width+match.x,match.dist});
                if(match.x==job.x && match.y==job.y) job.target=j;
            }
            if(job.target<0) throw std::runtime_error("NCSR HQ: matcher omitted anchor");
        }
    }
}

void assign(MatchPlan& plan,const Dictionaries& dict,const Vec& image,int width,int height,int block) {
    const int d=block*block;Vec patch(d),f(d);
    for(auto& job:plan.jobs) {
        pack_patch(patch.data(),d,image.data(),width,job.x,job.y,block,width,height);
        const float variance=feature(patch.data(),d,f.data());
        job.cls=variance<dict.flat_limit || dict.clusters==0 ? 0 : 1+classify(f.data(),d,dict.centers,dict.clusters);
    }
}

struct TileWorkspace {
    static constexpr std::size_t targets=128;
    ResourceVector<std::size_t> order;
    ResourceVector<int> positions;
    ResourceVector<double> taus;
    Vec packed,projected,target_codes,reconstructed,output,scratch;
    std::uint64_t requested=0,unique=0;
    int estimate_generation=-1,dictionary_generation=-1,cls=-1;
    void begin_generation(int estimate,int dictionary) {
        estimate_generation=estimate;dictionary_generation=dictionary;cls=-1;
        // Capacities are reused, but no coefficients survive an image or
        // dictionary generation. Position alone is never a valid cache key.
        positions.clear();projected.clear();requested=0;unique=0;
    }
};

void run_targets(Context& state,const Vec& image,Vec& num,Vec& den,int width,int height,
                 const SearchConfig& cfg,float sigma,const MatchPlan& plan,const Dictionaries& dict,
                 const Vec& fallback,const Vec& fallback_transpose,TileWorkspace& ws,
                 int estimate_generation,int dictionary_generation) {
    const int d=cfg.block*cfg.block;
    ws.begin_generation(estimate_generation,dictionary_generation);
    ws.scratch.resize(d*cfg.group+d+cfg.group);
    float* codes=ws.scratch.data();float* patch=codes+d*cfg.group;float* distances=patch+d;
    ws.positions.reserve(TileWorkspace::targets*cfg.group);
    for(std::size_t begin=0;begin<plan.jobs.size();begin+=TileWorkspace::targets) {
        const auto end=std::min(plan.jobs.size(),begin+TileWorkspace::targets);
        const auto count=end-begin;
        ws.order.resize(count);std::iota(ws.order.begin(),ws.order.end(),begin);
        std::sort(ws.order.begin(),ws.order.end(),[&](auto a,auto b){
            return plan.jobs[a].cls!=plan.jobs[b].cls?plan.jobs[a].cls<plan.jobs[b].cls:a<b;
        });
        ws.output.resize(count*d);ws.taus.resize(count*d);
        for(std::size_t start=0;start<count;) {
            ws.cls=plan.jobs[ws.order[start]].cls;
            auto stop=start+1;
            while(stop<count && plan.jobs[ws.order[stop]].cls==ws.cls) ++stop;
            const int targets=static_cast<int>(stop-start);
            ws.positions.clear();
            for(auto i=start;i<stop;++i) {
                const auto& job=plan.jobs[ws.order[i]];
                for(int j=0;j<job.count;++j) ws.positions.push_back(plan.neighbors[job.offset+j].position);
            }
            ws.requested+=ws.positions.size();
            std::sort(ws.positions.begin(),ws.positions.end());
            ws.positions.erase(std::unique(ws.positions.begin(),ws.positions.end()),ws.positions.end());
            const int unique=static_cast<int>(ws.positions.size());ws.unique+=unique;
            ws.packed.resize(d*unique);ws.projected.resize(d*unique);
            for(int j=0;j<unique;++j) {
                const int position=ws.positions[j];
                pack_patch(patch,d,image.data(),width,position%width,position/width,cfg.block,width,height);
                for(int row=0;row<d;++row) ws.packed[row*unique+j]=patch[row];
            }
            const bool use_fallback=dict.bases[ws.cls].empty();
            const auto& basis=use_fallback?fallback:dict.bases[ws.cls];
            const auto& transpose=use_fallback?fallback_transpose:dict.transposes[ws.cls];
            if(gemm_f32_f64(basis.data(),d,ws.packed.data(),unique,ws.projected.data(),unique,d,unique,d))
                throw std::runtime_error("NCSR HQ: projection failed");
            ws.target_codes.resize(d*targets);ws.reconstructed.resize(d*targets);
            for(auto i=start;i<stop;++i) {
                const auto ordinal=ws.order[i];const auto& job=plan.jobs[ordinal];
                for(int j=0;j<job.count;++j) {
                    const auto& neighbor=plan.neighbors[job.offset+j];
                    const int slot=static_cast<int>(std::lower_bound(ws.positions.begin(),ws.positions.end(),neighbor.position)-ws.positions.begin());
                    for(int row=0;row<d;++row) codes[j*d+row]=ws.projected[row*unique+slot];
                    distances[j]=neighbor.distance;
                }
                shrink_target(state,codes,d,job.count,job.target,sigma,distances,ws.taus.data()+(ordinal-begin)*d);
                for(int row=0;row<d;++row) ws.target_codes[row*targets+i-start]=codes[job.target*d+row];
            }
            if(gemm_f32_f64(transpose.data(),d,ws.target_codes.data(),targets,ws.reconstructed.data(),targets,d,targets,d))
                throw std::runtime_error("NCSR HQ: reconstruction failed");
            for(auto i=start;i<stop;++i)for(int row=0;row<d;++row)
                ws.output[(ws.order[i]-begin)*d+row]=ws.reconstructed[row*targets+i-start];
            start=stop;
        }
        // Class batching must not reorder image accumulation or diagnostics.
        for(std::size_t i=0;i<count;++i) {
            for(int row=0;row<d;++row) {state.tau_sum+=ws.taus[i*d+row];++state.tau_count;}
            const auto& job=plan.jobs[begin+i];
            aggregate_add(num.data(),den.data(),width,job.x,job.y,ws.output.data()+i*d,cfg.block,width,height,1.f);
        }
    }
}
}  // namespace

void denoise(const float* input,int width,int height,int stride,float* output,int output_stride,
             SearchConfig cfg,float sigma,int iterations,float delta,std::string& trace) {
    if(!input || !output || cfg.radius!=0 || cfg.block>16 || cfg.block<1 || cfg.step<1 || cfg.step>cfg.block ||
       cfg.group>32 || cfg.group<1 || cfg.bm_range<0 || width<cfg.block || height<cfg.block ||
       stride<width || output_stride<width || iterations<1 || iterations>12 ||
       !std::isfinite(sigma) || sigma<0 || !std::isfinite(delta) || delta<0 || delta>1)
        throw std::invalid_argument("NCSR HQ: invalid spatial configuration");
    const int pixels=checked_int(checked_mul(width,height));
    Vec estimate(pixels),noisy(pixels),num(pixels),den(pixels);
    const Vec fallback=dct_basis(cfg.block);
    Vec fallback_transpose(fallback.size());
    const int d=cfg.block*cfg.block;
    for(int row=0;row<d;++row)for(int col=0;col<d;++col) fallback_transpose[row*d+col]=fallback[col*d+row];
    for(int y=0;y<height;++y) std::copy_n(input+static_cast<std::size_t>(y)*stride,width,noisy.data()+y*width);
    for(float value:noisy) if(!std::isfinite(value)) throw std::invalid_argument("NCSR HQ: nonfinite input");
    estimate=noisy;
    Context state;state.input_sigma=sigma;state.match_sigma=sigma;
    MatchPlan plan; Dictionaries dictionaries;TileWorkspace workspace;std::ostringstream log;
    log.precision(10);log<<"{\"model\":\"clustered-pca-hq\",\"iterations\":[";
    for(int iteration=0;iteration<iterations;++iteration) {
        double training_seconds=0,matching_seconds=0,classification_seconds=0;
        if(iteration%3==0) {
            auto start=Clock::now();dictionaries=train(estimate,width,height,cfg.block,state.match_sigma);
            training_seconds=elapsed(start);
            start=Clock::now();match_all(plan,estimate,width,height,cfg);matching_seconds=elapsed(start);
            start=Clock::now();assign(plan,dictionaries,estimate,width,height,cfg.block);classification_seconds=elapsed(start);
        }
        if(iteration>0) iter_regularize(estimate.data(),noisy.data(),pixels,delta);
        const float effective_sigma=iteration>0?residual_sigma(sigma,estimate,noisy):sigma;
        std::fill(num.begin(),num.end(),0.f);std::fill(den.begin(),den.end(),0.f);
        const auto start=Clock::now();const double old_tau=state.tau_sum;const auto old_count=state.tau_count;
        run_targets(state,estimate,num,den,width,height,cfg,effective_sigma,plan,dictionaries,fallback,fallback_transpose,
                    workspace,iteration,iteration/3);
        aggregate_finish(estimate.data(),num.data(),den.data(),estimate.data(),width,height,width,width);
        for(float value:estimate) if(!std::isfinite(value)) throw std::runtime_error("NCSR HQ: nonfinite output");
        const auto tau_count=state.tau_count-old_count;
        if(iteration) log<<",";
        log<<"{\"sigma8\":"<<effective_sigma*255<<",\"training_seconds\":"<<training_seconds
           <<",\"matching_seconds\":"<<matching_seconds<<",\"classification_seconds\":"<<classification_seconds
           <<",\"filtering_seconds\":"<<elapsed(start)<<",\"mean_tau\":"<<(tau_count?(state.tau_sum-old_tau)/tau_count:0)
           <<",\"dictionary_clusters\":"<<dictionaries.clusters<<",\"training_samples\":"<<dictionaries.training_samples
           <<",\"match_storage_bytes\":"<<plan.jobs.capacity()*sizeof(Neighborhood)+plan.neighbors.capacity()*sizeof(Neighbor)
           <<",\"requested_projections\":"<<workspace.requested<<",\"unique_projections\":"<<workspace.unique<<"}";
        if((iteration+1)%3==0) state.match_sigma=effective_sigma;
    }
    log<<"]}";trace=log.str();
    for(int y=0;y<height;++y) std::copy_n(estimate.data()+y*width,width,output+static_cast<std::size_t>(y)*output_stride);
}
}  // namespace nss::ncsr_hq
