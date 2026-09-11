#pragma once
#include "hwy/targets.h"
#include "nss/backend.hpp"
#include <cstring>
#include <iostream>

inline bool hq_test_target(int argc,char** argv) {
    if(argc==1) return true;
    const auto target=std::strcmp(argv[1],"avx2")==0?HWY_AVX2:
                      std::strcmp(argv[1],"avx3")==0?HWY_AVX3:0;
    const auto before=nss::backend_caps();
    if(!target || !(before.compiled_targets&before.runtime_targets&target)) return false;
    hwy::SetSupportedTargetsForTest(target);
    const auto selected=nss::backend_caps();
    if(selected.selected_target!=target) return false;
    std::cout<<"verified Highway target="<<selected.target_name<<" lanes="<<selected.float_lanes<<"\n";
    return true;
}
