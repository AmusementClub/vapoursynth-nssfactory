#include "nss/cpu_nlh.hpp"
#include "nss/cpu_nlh_full.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

static void require(bool yes,const char* message) { if(!yes) throw std::runtime_error(message); }
static std::vector<double> haar(int n) {
    std::vector<double> h(n*n);
    for(int j=0;j<n;++j) h[j]=1/std::sqrt(double(n));
    int row=1;
    for(int width=n;width>=2;width/=2) for(int start=0;start<n;start+=width) {
        for(int j=0;j<width;++j) h[row*n+start+j]=(j<width/2?1:-1)/std::sqrt(double(width));
        ++row;
    }
    return h;
}
int main() {
    try {
        for(int n:{1,2,4,8,16,32,64}) {
            std::vector<float> a(n),b(n),restored(n);
            for(int i=0;i<n;++i) a[i]=float(std::sin(i*.29));
            nss::haar1d(a.data(),b.data(),n);
            const auto h=haar(n);
            for(int i=0;i<n;++i) {
                double expected=0;
                for(int j=0;j<n;++j) expected+=h[i*n+j]*a[j];
                require(std::abs(b[i]-expected)<2e-5,"Haar matrix oracle");
            }
            nss::ihaar1d(b.data(),restored.data(),n);
            for(int i=0;i<n;++i) require(std::abs(restored[i]-a[i])<2e-5,"Haar inverse");
        }
        for(int rows:{2,4,8,16}) for(int columns:{1,2,4,8,16,32,64}) {
            std::vector<float> matrix(rows*columns);
            for(int j=0;j<columns;++j) for(int i=0;i<rows;++i)
                matrix[i+j*rows]=float(std::sin((i+7*j)*.117)+.2*std::cos((3*i+j)*.193));
            const auto original=matrix;
            const auto left=haar(rows),right=haar(columns);
            nss::nlh_haar2d(matrix.data(),rows,columns,false);
            for(int j=0;j<columns;++j) for(int i=0;i<rows;++i) {
                double expected=0;
                for(int c=0;c<columns;++c) for(int r=0;r<rows;++r)
                    expected+=left[i*rows+r]*original[r+c*rows]*right[j*columns+c];
                require(std::abs(matrix[i+j*rows]-expected)<4e-5,"two-sided Haar matrix oracle");
            }
            nss::nlh_haar2d(matrix.data(),rows,columns,true);
            for(std::size_t i=0;i<matrix.size();++i)
                require(std::abs(matrix[i]-original[i])<4e-5,"two-sided Haar inverse");
        }
        constexpr int m=19,n=16,q=4,lda=23;
        std::vector<float> a(lda*n,-777),ref(lda*n,-777);
        for(int j=0;j<n;++j) for(int i=0;i<m;++i) a[i+j*lda]=ref[i+j*lda]=.2f;
        const auto saved=a;
        nss::NlhWorkspace work;
        nss::NlhGroupOptions options{q,1,false,1,2,.5};
        nss::nlh_filter_full(a.data(),nullptr,m,n,lda,options,nullptr,work);
        require(a==saved,"NLH input must be immutable");
        double count=0;
        for(double x:work.denominator) count+=x;
        require(count==m*n*q,"all q rows must aggregate");
        for(double x:work.numerator) require(x==0,"linear hard threshold must not exempt DC");
        // Scaling samples and sigma together must preserve the hard mask.
        // A sigma-squared threshold fails this check at the larger scale.
        options.sigma=.5;
        nss::nlh_filter_full(a.data(),nullptr,m,n,lda,options,nullptr,work);
        const auto base_num=work.numerator,base_den=work.denominator;
        auto scaled=a;
        for(int j=0;j<n;++j) for(int i=0;i<m;++i) scaled[i+j*lda]*=4;
        options.sigma=2;
        nss::nlh_filter_full(scaled.data(),nullptr,m,n,lda,options,nullptr,work);
        require(work.denominator==base_den,"linear scaling preserves contribution counts");
        for(std::size_t i=0;i<work.numerator.size();++i)
            require(std::abs(work.numerator[i]-4*base_num[i])<2e-6,"linear sigma/sample scale covariance");
        options.sigma=.5;options.hard_strength=2;
        nss::nlh_filter_full(a.data(),nullptr,m,n,lda,options,nullptr,work);
        for(double x:work.numerator) require(x==0,"hard_strength must multiply the linear threshold");
        options.hard_strength=1;
        options.sigma=.5;options.wiener=true;
        nss::nlh_filter_full(a.data(),ref.data(),m,n,lda,options,nullptr,work);
        const double dc=double(.2f)*std::sqrt(double(q*n));
        const double gain=dc*dc/(dc*dc+.25*.25);
        for(std::size_t i=0;i<work.numerator.size();++i)
            require(std::abs(work.numerator[i]/work.denominator[i]-double(.2f)*gain*gain)<2e-6,"two Wiener gains without DC exemption");
        options.sigma=0;
        nss::nlh_filter_full(a.data(),ref.data(),m,n,lda,options,nullptr,work);
        for(std::size_t i=0;i<work.numerator.size();++i)
            require(work.numerator[i]/work.denominator[i]==double(.2f),"zero noise bypass");
        bool rejected=false; options.sigma=std::numeric_limits<double>::infinity();
        try { nss::nlh_filter_full(a.data(),ref.data(),m,n,lda,options,nullptr,work); }
        catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"nonfinite sigma");
        std::puts("test_nlh independent Haar, linear hard, scale covariance, full aggregation and repeated Wiener ok");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
