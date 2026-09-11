#include "nss/cpu_twsc_full.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool yes, const char* message) { if (!yes) throw std::runtime_error(message); }
int main() {
    try {
        // Analytic independent optimum for a diagonal dictionary, including a
        // zero spectral direction. This is the FULL objective's 1/2 factor.
        constexpr int m = 6, n = 4, r = 4;
        double d[m*r]{}, s[r]{1.1, 2, 0, 4}, y[m*n], precision[m]{2,3,4,5,6,7};
        float sigma[n]{.3f,.4f,.5f,.6f};
        for (int i=0;i<r;++i) d[i+i*m]=1;
        for (int j=0;j<n;++j) for (int i=0;i<m;++i) y[i+j*m]=.2*std::sin(.3*(1+i+7*j));
        double output[m*n]; nss::TwscWorkspace w;
        auto status=nss::twsc_solve(y,d,s,m,n,r,precision,sigma,{1000,2,1.01,1e-10},output,w);
        require(status.converged,"diagonal objective must converge");
        require(status.sylvester_residual<=1e-10,"Sylvester residual");
        for (int j=0;j<n;++j) for (int i=0;i<m;++i) {
            const double expected=i<r && s[i]>0 ? std::copysign(std::max(std::abs(y[i+j*m])-double(sigma[j])/(2*precision[i]*s[i]),0.0),y[i+j*m]):0;
            require(std::abs(output[i+j*m]-expected)<2e-7,"full weighted objective mismatch");
        }
        // A finite cap is not a convergence claim.
        status=nss::twsc_solve(y,d,s,m,n,r,precision,sigma,{1,.5,1.1,1e-12},output,w);
        require(status.iterations==1 && !status.converged,"finite ADMM cap status");
        // Constant group centering has rank zero; restore each row exactly.
        constexpr int rows=48, columns=8, lda=53;
        std::vector<float> group(lda*columns,-987), row_sigma(rows,.01f), col_sigma(columns,.01f), weights(columns);
        for (int j=0;j<columns;++j) for (int i=0;i<rows;++i) group[i+j*lda]=float(i)/128;
        const auto original=group;
        status=nss::twsc_filter_full(group.data(),rows,columns,lda,row_sigma.data(),col_sigma.data(),weights.data(),{},w);
        require(group==original,"rank-zero reconstruction and padding");
        require(status.converged && status.iterations==1,"rank-zero status");
        nss::TwscSolverOptions invalid_options{0,.5,1.1,1e-6};
        nss::TwscFullBatchItem invalid_batch{group.data(),rows,columns,lda,row_sigma.data(),col_sigma.data(),weights.data(),&invalid_options,&w,nullptr};
        bool invalid_rejected=false;
        try { nss::twsc_filter_full_batch(&invalid_batch,1); }
        catch (const std::invalid_argument&) { invalid_rejected=true; }
        require(invalid_rejected,"invalid ADMM parameters cannot bypass checks at rank zero");
        // Shared source limits remain independent of the large TWSC lane.
        std::vector<float> a(49*70);
        for (int j=0;j<70;++j) for (int i=0;i<49;++i) a[i+j*49]=float(std::sin((i+1)*.13)*std::cos((j+1)*.17));
        bool fallback=false;
        require(nss::twsc_svd(a.data(),49,70,49,w,fallback),"wide SVD");
        require(!nss::twsc_svd(a.data(),49,257,49,w,fallback),"SVD upper bound");
        col_sigma[0]=std::numeric_limits<float>::quiet_NaN();
        bool rejected=false;
        try { nss::twsc_filter_full(group.data(),rows,columns,lda,row_sigma.data(),col_sigma.data(),weights.data(),{},w); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"nonfinite column sigma must fail even at rank zero");
        // Reusable workspaces must release every budgeted allocation.
        auto budget=std::make_shared<nss::ResourceBudget>(1024*1024);
        { nss::ResourceScope scope(budget); nss::TwscWorkspace owned;
          require(nss::twsc_svd(a.data(),49,70,49,owned,fallback),"budgeted SVD"); }
        require(budget->snapshot().owned==0,"workspace leak");
        std::puts("test_twsc full objective, finite cap, rank-zero, wide SVD and resources ok");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
