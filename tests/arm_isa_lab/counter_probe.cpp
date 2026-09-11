#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <cstring>
#include <cstdio>
extern "C" void nss_neon_instruction_loop(unsigned long count);
extern "C" void nss_sve_instruction_loop(unsigned long count);
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    if(std::strcmp(argv[1],"sve")==0) {
        if(!(getauxval(AT_HWCAP)&HWCAP_SVE))return 77;
        nss_sve_instruction_loop(10000000);
    } else if(std::strcmp(argv[1],"neon")==0) {
        nss_neon_instruction_loop(10000000);
    } else return 2;
    std::puts("40000000 vector FMLA instructions completed");
}
