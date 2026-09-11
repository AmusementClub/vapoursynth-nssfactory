#include <arm_sve.h>
#include <cstdio>

#ifdef NSS_SVE_FIXED_TYPES
using Stored = svfloat32_t __attribute__((arm_sve_vector_bits(128)));
#else
using Stored = svfloat32_t;
#endif

int main() {
    Stored rows[8];
    for (int i=0;i<8;++i) rows[i]=svdup_n_f32(float(i));
    const auto pg=svptrue_b32();
    const auto sum=svadd_f32_x(pg,rows[3],rows[4]);
    const float result=svaddv_f32(pg,sum);
    std::printf("lanes=%lu sum=%.1f stored_bytes=%zu\n",svcntw(),result,sizeof(rows));
    return result==28.f?0:1;
}
