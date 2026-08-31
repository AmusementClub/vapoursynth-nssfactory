#pragma once

#include <VapourSynth4.h>

#include <string>

namespace nss {

bool is_const_32f(const VSVideoInfo& vi);
bool same_video(const VSVideoInfo& a, const VSVideoInfo& b);
int plane_width(const VSVideoInfo& vi, int plane);
int plane_height(const VSVideoInfo& vi, int plane);

float map_float(const VSAPI* vsapi, const VSMap* in, const char* key, float def, int* err = nullptr);
int map_int(const VSAPI* vsapi, const VSMap* in, const char* key, int def, int* err = nullptr);
void map_float_array(const VSAPI* vsapi, const VSMap* in, const char* key, float* out, int n, float def);
void map_int_array(const VSAPI* vsapi, const VSMap* in, const char* key, int* out, int n, int def);
void map_inherit_int(const VSAPI* vsapi, const VSMap* in, const char* key, int* out, int n, int def);

}  // namespace nss
