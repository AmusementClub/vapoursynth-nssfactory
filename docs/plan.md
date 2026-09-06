# vapoursynth-nssfactory 开发计划

状态：CPU 综合优化已有七个算法通过正式 C4 gate；仅 BM3D 尚未达到 `1.25x`，整个 campaign 继续保持未通过。
本文是从上往下的施工图：目标 → 约束 → 架构 → 模块 → 接口 → 阶段。后续实现以本文为准；改决策先改本文。

## 当前状态（2026-09-01）

- 原目录 `/Users/owen/Documents/vapoursynth-nssfactory` 保持基线提交 `223003e`，未修改。
- 活动 checkout `/Users/owen/Documents/vapoursynth-nssfactory-work` 的历史基线为 `310adb0`；CPU 优化改动仍在未提交工作树中，按要求不在本轮创建 commit。
- 活动 checkout 本机 Release、VapourSynth host 和 ASan/UBSan 构建均成功；三套 CTest 都是 **14/14 passed**。撤除 NLM 临时 benchmark 探针后又执行了一次干净 Release 重建，仍为 **14/14 passed**。ASan 使用 `halt_on_error=1`/UBSan stacktrace 检查，macOS 不启用不受支持的 leak detector；ARM 主机可加载 `libnss.dylib` 并列出全部入口，但不执行要求 AVX2 的滤镜路径。
- 远端开发回归主机 `owen@192.168.50.3`（5950X）Release CTest 为 **14/14 passed**。NLM、BM3D Basic→Final、WNNM `radius=1`、TWSC、MCWNNM、NCSR、NLH 和 LSSC 均以独立重建节点完成 1/2-thread frame hash smoke；这些结果是开发回归，不是正式性能 gate。
- 已在 `dsmvc-avx512-pmu/us-central1-a` 手动创建带 `STANDARD` PMU 的 `c4-highcpu-2` Spot 实例 `nss-c4-topdown-20260831125510`，通过 SSH 完成八个 CPU 算法的 1080p Topdown L1/L2/L3 与 `perf record`。硬件计数非零、八份 `perf.data` 均可解析且 Lost Samples 为 0；本次手动流程不依赖 Policy Troubleshooter API。实例暂时保留用于紧接着的 paired A/B，所有实验结束后必须手动删除并确认不存在。
- 正式 paired C4 已确认 WNNM batched SVD 中位数约 **1.2067x**（门槛 1.15x）、NCSR 约 **1.3075x**（门槛 1.15x）、NLM 15 对中位数 **1.334901x**（范围 1.288994x--1.342539x，门槛 1.30x）；NLM 15 对输出 hash 全部相同。
- TWSC 的 Highway U/S-only SVD 已通过正式 C4：15 对中位数 **1.400585688480x**，6 组合法配置最低单对 **1.021764648674x**，相对 baseline PSNR `146.541 dB`。MCWNNM 的 Gram spectral-shrink 已通过正式 C4：15 对中位数 **1.577800007075x**，12 组合法配置最低单对 **0.992922490995x**，PSNR `150.049 dB`，repeat bit-exact。
- NLH 的 Highway q4/n16 Wiener、vector-only 低 4-lane Haar、q4 fixed top-3 PixelMatch 和 masked tail 已通过正式 C4：7 对中位数 **1.402357948x**（范围 1.401808564x--1.403821488x，门槛 1.30x）。相对 baseline 的最大绝对误差 `7.15e-7`、PSNR `142.58 dB`，candidate 重复逐元素完全一致；24 组合法 block/group/q 矩阵最差 ratio `0.994732259x`。
- LSSC 的 AVX-512 Highway 2-vector×8-column `GemmNN` 和 fused ISTA update/group-soft/check 已通过正式 C4：7 对中位数 **1.331358620x**（范围 1.326825634x--1.340783378x，门槛 1.15x）。输出 2073600 个 float 与 baseline 完全相同；全部 7 个合法 block/step 组合最差 ratio 为 `1.166268191x`。
- 已撤回的性能实验：candidate-lane matcher 在所测算法上全面回退；TWSC 沿用同一 batched SVD 仅约 **1.094x**；MCWNNM SVD 预筛约 **0.824x**；NLM horizontal prefix 与全向量 exp underflow 分支均约 **0.99x**。这些失败实现均不在当前候选中。
- BM3D 当前正式真实图 paired speedup 为 **1.109129869x**，低于 `1.25x`。基于 9 张真实 1080p 图、18,360 个 reference block 和 4,024,251 个候选的 FMA/reduction-tree 匹配模型确认：最强的纯 first4 可实现调度只有 **15.0301%** SSD 行工作节省；即使把尚未证明通用浮点精确、且必须读取 tail 的 tail-DC 当作零成本，中心优先调度也只有 **18.1883%**，低于过 gate 所需的 **19.1205%**。first4 精确剪枝路线关闭，证据见 `artifacts/c4/bm3d-pruning-20260901/bm-first4-v7-summary.txt`。

### CPU 综合优化状态矩阵

| Checkpoint | 当前状态 | 证据/剩余边界 |
|---|---|---|
| W0 正确性基线与观测协议 | 部分完成 | corrected SVD、固定 seed/JSON v2 已加入；尚无独立 checkpoint commit |
| W1 matcher 与 batch 核心 | 完成（实现级） | scalar differential、稳定排序、有限值回退和有界 ordered commit 已通过；当前 wrapper 仍是分桶后逐项 kernel，不是跨 lane SIMD |
| W2 BM3D 主流水线 | 部分完成 | Basic/Final、radius 0/1 和 batch host 已接入，Basic→Final 插件 smoke/hash 通过；正式非默认矩阵与 C4 `>=1.25x` 未完成 |
| W3 WNNM/SVD | C4 gate 通过 | batched host、SVD 质量测试、`radius=1` smoke/hash 通过；正式 paired C4 中位数约 `1.2067x`，超过 `1.15x` |
| W4 NLM lane | C4 gate 通过 | fused Highway Welsch、fixed-s horizontal reduction、zero-weight skip、compact temporal/base-row accumulation 已保留；15 对中位数 `1.334901x`，hash 全同 |
| W5 TWSC/MCWNNM/NCSR/NLH | C4 gate 通过 | TWSC `1.400585688480x`、MCWNNM `1.577800007075x`、NCSR 约 `1.3075x`、NLH `1.402357948x`；各自合法配置矩阵最低单对均不低于 `0.98x` |
| W6 LSSC prepared pipeline | C4 gate 通过 | prepared context、prefix members、workspace-only 临时区、2×8 GemmNN 与 fused ISTA 已完成；正式中位数 `1.331358620x`，7 组合法矩阵最差 `1.166268191x`，输出逐 float 相同 |
| W7 正式 C4 campaign 与收尾 | 进行中 | 八算法 Topdown 已完成；WNNM、TWSC、MCWNNM、NCSR、NLM、NLH、LSSC 已通过；仅 BM3D `1.109129869x` 未达到 `1.25x` |

本轮 batch 接入顺序冻结为 **BM3D → WNNM → TWSC → MCWNNM → NCSR → NLH → LSSC**。七个非 BM3D 算法已经通过各自 C4 gate；BM3D 通过前不能宣布整个 CPU 综合优化完成。CUDA/Vulkan 和 GPU 原始阶段不计入本轮完成定义。

---

## 0. 一句话

一个 VapourSynth **API4** 插件厂：NSS 类降噪共用 host / 搜索 / 聚合协议，CPU 用 **C++20 + Highway**，GPU 用 **互相独立的 CUDA 与 Vulkan** 流水线。运行期 **纯 CPU xor 纯 CUDA xor 纯 Vulkan**，禁止算法级设备往返，也禁止 CUDA↔Vulkan 互操作。

第一版平台：**Linux x86-64，最低 AVX2**。

---

## 1. 已锁定决策

| # | 决策 | 不选 | 理由 |
|---|---|---|---|
| D1 | VS **API4**（`VapourSynthPluginInit2` / `VSHelper4.h`） | API3 | 新插件没有兼容旧 ABI 的负担 |
| D2 | CPU SIMD = **Google Highway**，动态派发 | ISPC、VCL、手写 AVX2-only、Halide | 与 C++ 同编译器；Zen4 有 `AVX3_ZEN4`；无第二套编译器 |
| D3 | 运行期设备封闭：一帧内 search→filter→agg 全在同一设备 | CPU 搜块 / GPU SVD 等混合 | PCIe + 小缓冲 + 线程同步会把小矩阵优势吃光 |
| D4 | 三个产物：`libnss.so`、可选 `libnss_cuda.so`、可选 `libnss_vk.so` | 单一 `.so` + `backend=`；CUDA↔Vulkan 互操作 | 无对应驱动的机器不能被拖死；用户用命名空间选设备 |
| D5 | WNNM SVD = **自研 64×8 tall-skinny**（QR + 8×8 Jacobi） | MKL/AOCL 硬依赖 | 默认 `block_size=8, group_size=8` 上厂商 LAPACK 调用税高于计算 |
| D6 | 内部工作格式 **f32** | 核内混 u8/u16 | 与 BM3D/WNNM 参考一致；整数输入后期边界转换 |
| D7 | VS 线程 = CPU 并行度；MKL/OpenBLAS/Highway 内层不再开线程 | 滤镜内 OpenMP | 避免和 `fmParallel` 双重订阅 |
| D8 | 时间域聚合是 **独立阶段** `VAggregate`。GPU 树上该阶段必须是 **独立 device kernel**，中间量（num/den 栈）留在 device，只 DTOH 最终帧。`BM3Dv2` 只是 `BM3D`+`VAggregate` 的语法糖 | 把聚合并进协同滤波核；GPU BM3D 出胖中间帧再接 **CPU** `VAggregate` | 并进 BM 核会打乱占用率/访存；CPU 聚合要搬 `(2R+1)×2` 平面，这正是 bm3dcuda 上 GPU `VAggregate` 大提速所砍掉的路径。两段 API 也才能把 aggregated basic 交给 Final 的 `ref` |
| D9 | CPU/GPU **不保证 bitwise identical** | 跨设备金标准 | 与 BM3DCUDA 现状一致；对拍 PSNR/残差 |
| D10 | LSSC / NCSR / NLH **不进第一版**。MCWNNM / TWSC 列入第二波（§14），不与 NLM/BM3D/WNNM 并行开工 | 六算法同时开工 | 无稳定 VS 参考；先把 NLM/BM3D/WNNM 的 factory 跑通 |
| D11 | GPU 先 **CUDA**，后 **Vulkan**；每条 GPU 树 internally 封闭 | 先写 Vulkan、两套 GPU 并行开工、用 SYCL/Halide 统一 | CUDA 有 nlm-cuda / BM3DCUDA 可对拍；Vulkan 是移植不是探索 |
| D12 | 第一轮 GPU 仍吃 **CPU VSFrame**（HTOD→核→DTOH） | 一上来接 VS R80 GPU VideoNode | 核心 GPU 路径未稳定；先把计算核做对 |

参考实现（行为对拍，不复制其工具链）：

- [vs-nlm-ispc](https://github.com/AmusementClub/vs-nlm-ispc) — NLM 算法与参数
- [VapourSynth-WNNM](https://github.com/WolframRhodium/VapourSynth-WNNM) — BM + WNNM + V-BM3D 搜索语义
- [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) — GPU 封闭流水线、CPU AVX2 BM3D、参数名

---

## 2. 目标与非目标

### 2.1 目标

1. 用户脚本里用 `core.nss.NLM` / `nss.BM3D` / `nss.BM3Dv2` / `nss.WNNM` 做 CPU 降噪。
2. 可选 `core.nss_cuda.*` / `core.nss_vk.*` 做 GPU 降噪，参数名与 CPU 对齐，数据不回 host 做协同滤波。
3. block matching、patch 打包、加权写回在 CPU 树内只写一次，三个算法插不同 filter。
4. Linux x86 上 Highway 运行时选择 AVX2 / AVX3 / AVX3_ZEN4，一份源码。
5. 与参考插件在约定参数下质量可对拍（见 §8）。

### 2.2 非目标（第一版明确不做）

- Windows / macOS / ARM 发行
- ISPC、Halide、SYCL、HIP、Metal、Mojo
- 一帧内 CPU+GPU，或 CUDA+Vulkan
- 运行时切换 backend
- GPU WNNM（更后期：device 端 batched 小 SVD，CUDA/Vulkan 各写一份）
- 用一层抽象把 CUDA 和 Vulkan 写成同一份 kernel
- NLH / LSSC / NCSR / MCWNNM / TWSC（第一版不做；第二波见 §14）
- 与参考插件 bitwise 一致
- 可变分辨率 / 可变格式 clip
- 在核内做色彩空间（RGB↔OPP 留给脚本或后续 `nss.RGB2OPP`）

---

## 3. 分层架构（从上往下）

```
脚本 / vs-jetpack
        │
        ▼
┌───────────────────────────────────────┐
│  VS API4 host                         │  创建期绑定设备；getFrame 只调度
│  格式检查、requestFrame、workspace     │
└───────────────┬───────────────────────┘
                │
     ┌──────────┼──────────────────┐
     ▼          ▼                  ▼
┌──────────┐ ┌────────────┐ ┌────────────┐
│ CpuPipe  │ │ CudaPipe   │ │ VulkanPipe │  编译期三棵树，禁止互调
│ libnss   │ │ nss_cuda   │ │ nss_vk     │
└────┬─────┘ └─────┬──────┘ └─────┬──────┘
     ▼             ▼              ▼
 Highway        CUDA 核       Vulkan 计算着色器
 + 小 SVD     （整条 device） （整条 device）
```

规则：

- **语言边界可以对齐 CPU 地址空间**（Highway ↔ 小 SVD ↔ host），**不能跨 PCIe，也不能跨 CUDA context ↔ VkDevice**。
- CUDA 与 Vulkan **不共享 kernel 源**。共享物只有：`params` 默认值、距离定义、测试向量、文档。
- NLM 与 BM 类算法 **不共用 inner kernel**（整帧距离图 vs 每 reference 一个 8×8×K）。
- 同一算法的 GPU 实现顺序：先 CUDA（有参考、调试工具成熟），再把**已验证的算法**平移到 Vulkan，而不是两边同时探索。

---

## 4. 仓库布局

第一版按「能编过的空厂」一次铺齐目录，算法核按阶段填入。

```text
vapoursynth-nssfactory/
  CMakeLists.txt
  cmake/
    FindVapourSynth.cmake
    Highway.cmake                 # FetchContent，钉版本
  docs/
    plan.md                       # 本文
  include/nss/
    version.hpp
    params.hpp                    # 默认参数、距离枚举，CPU/GPU 都可 include
    plane.hpp                     # 非拥有视图：ptr, width, height, stride
  src/
    host/
      plugin.cpp                  # VapourSynthPluginInit2
      validate.cpp                # clip/格式/参数
      workspace.cpp               # thread_id → 对齐缓冲
      filter_nlm.cpp
      filter_bm3d.cpp
      filter_wnnm.cpp
    cpu/
      dispatch.hpp                # HWY_EXPORT / HWY_DYNAMIC_DISPATCH 包装
      nlm/
        nlm.cpp                   # Highway 目标函数
      bm/
        match.cpp                 # 窗口 + predictive search + partial_sort（标量 C++）
        ssd.cpp                   # Highway SSD
        pack.cpp                  # im2col / 中心化
        dct8.cpp                  # 8×8 DCT（Highway）
        haar.cpp                  # 预留
        agg.cpp                   # 加权 col2im（Highway）
      wnnm/
        svd_tiny.cpp              # QR + Jacobi，无 LAPACK
        shrink.cpp                # 加权奇异值 + 重建
    cuda/                         # NSS_ENABLE_CUDA；Phase 5 才填
      plugin.cpp
      nlm.cu
      bm3d.cu
    vulkan/                       # NSS_ENABLE_VULKAN；Phase 6 才填
      plugin.cpp
      shaders/                    # GLSL，构建期编成 SPIR-V
        nlm.comp
        bm3d.comp
      nlm.cpp
      bm3d.cpp
  tests/
    CMakeLists.txt
    ref/                          # 禁止提交大视频；放生成脚本与哈希
    test_svd.cpp
    test_nlm_cpu.cpp
    test_bm3d_cpu.cpp
    vs/                           # vspipe 脚本，可选
  .github/workflows/linux.yml
```

插件标识：

| | CPU | CUDA | Vulkan |
|---|---|---|---|
| id | `com.nssfactory.nss` | `com.nssfactory.nss_cuda` | `com.nssfactory.nss_vk` |
| namespace | `nss` | `nss_cuda` | `nss_vk` |
| 库名 | `libnss.so` | `libnss_cuda.so` | `libnss_vk.so` |

---

## 5. 对外滤镜（脚本层）

参数名对齐参考插件，降低 vs-jetpack / 现有脚本迁移成本。第一版全部要求 **常量格式、32-bit float**；平面独立处理（BM3D `chroma` 第一版不做）。

### 5.1 `nss.NLM`

对齐 `nlm_ispc.NLMeans`：

```text
clip:vnode; d:int:opt; a:int:opt; s:int:opt; h:float:opt;
channels:data:opt; wmode:int:opt; wref:float:opt; rclip:vnode:opt;
```

默认：`d=1, a=2, s=4, h=1.2, channels="AUTO", wmode=0, wref=1.0`。

### 5.2 `nss.BM3D` / `nss.BM3Dv2`

对齐 BM3DCUDA 常用子集：

```text
clip:vnode; ref:vnode:opt; sigma:float[]:opt;
block_step:int[]:opt; bm_range:int[]:opt; radius:int:opt;
ps_num:int[]:opt; ps_range:int[]:opt;
```

- 无 `ref` → Basic（hard threshold）
- 有 `ref` → Final（Wiener）
- `radius>0` 时 `BM3D`/`WNNM` 输出 **未聚合的中间量**（与现 bm3dcuda 相同的胖帧布局：每个时间偏移的 num/den），再接本插件的 `nss.VAggregate` / `nss_cuda.VAggregate` / `nss_vk.VAggregate`
- `BM3Dv2` 只是创建期把上面两步串起来的糖，**内部仍是两次核**（协同滤波 → 聚合），不是把聚合写进 BM kernel
- **不**调用 HOVE `bm3d.VAggregate`；GPU 路径 **不**把中间量拉回 CPU 再聚
- `group_size` 第一版 **固定 8**（与 bm3dcpu 一致，DCT 好写）
- `block_size` 第一版 **固定 8**

### 5.3 `nss.VAggregate`

```text
clip:vnode; src:vnode; radius:int:opt; planes:int[]:opt;
```

`clip` 为未聚合中间量，`src` 为原始（拷属性、未处理平面）。语义对齐 bm3dcuda 自带 `VAggregate`（复制 padding；与 HOVE 的 zero padding 不同，文档写明）。

CPU：普通平面归约。
GPU：独立 compute kernel，输入中间量在 device 上，输出最终帧。

### 5.4 `nss.WNNM`

对齐 WNNM：

```text
clip:vnode; sigma:float[]:opt; block_size:int:opt; block_step:int:opt;
group_size:int:opt; bm_range:int:opt; radius:int:opt;
ps_num:int:opt; ps_range:int:opt; residual:int:opt;
adaptive_aggregation:int:opt; rclip:vnode:opt;
```

默认与 WolframRhodium 插件相同（含加速过的 `block_size/step/group_size`）。
`block_size≠8` 时小 SVD 走通用 Jacobi（仍无 LAPACK）；测试主路径仍是 8。

时间域搜索语义跟 **WNNM / 旧 V-BM3D**（不要 BM3DCUDA 那种可能重复命中的加速搜索）。CPU BM3D 与 WNNM **共用这套搜索**。

---

## 6. 对内 C++ 边界

### 6.1 视图与参数（无设备）

```cpp
// include/nss/plane.hpp
struct PlaneView {
    const float* ptr;
    float*       mut;      // 只在输出平面非空
    int width, height, stride;
};

enum class Distance { SSD /* 第一版只做这个 */ };

struct SearchConfig {
    int block = 8;
    int step  = 8;
    int group = 8;
    int bm_range = 7;
    int radius = 0;
    int ps_num = 2;
    int ps_range = 4;
};
```

`params.hpp` 只放默认值与校验范围，不放 VS 类型。

### 6.2 CPU 流水线（同进程、同地址空间）

```text
host getFrame
  → 填 PlaneView（含 temporal 指针数组，长度 2R+1）
  → CpuNlm::run            或
  → CpuBm::match  → Filter → CpuBm::aggregate
```

- `CpuBm::match`：C++ 控制流（窗口、predictive search、`partial_sort`）+ Highway SSD。
- `Bm3dHard` / `Bm3dWiener` / `WnnmShrink`：只吃已打包的 group 矩阵。
- NLM **不**走 `CpuBm`。

Highway 约定：

- 核函数写在 `HWY_NAMESPACE` 里，文件末 `HWY_ONCE` 导出。
- 对外 C++ 包装只暴露 `void nlm_distance_luma_f32(...)` 这种非模板名。
- 动态派发目标：**AVX2 必选**；AVX3 与 AVX3_ZEN4 随 CPUID。不编译 SSE4 回退（第一版 min AVX2，启动时 CPUID 不够则 `mapSetError`）。
- `HWY_DISABLED_TARGETS` 关掉 AVX2 以下。

小 SVD 约定：

- 输入列主、lda 对齐 8。
- 输出 thin U（64×k）、S（k）、Vt（k×8），k ≤ min(m,n)。
- 失败（NaN/不收敛）返回错误码，host 对该 group 跳过协同滤波并计入计数器（debug 帧属性可选）。

### 6.3 GPU 流水线（CUDA 与 Vulkan 各一份）

`src/cuda/**` 与 `src/vulkan/**` 不得 `#include` `src/cpu/**`，彼此也不得互 include。

共同约定（两套后端各自实现，不要过早抽公共 GPU 基类）：

- 创建期分配 device 缓冲；`getFrame`：HTOD → 核 → DTOH **最终帧**。
- 并行度：`num_streams` 默认 2，池挂在滤镜实例上，**不要** `thread_id → 无限 stream/command buffer`。
- 时间域：协同滤波核把 num/den 写到 **device 中间缓冲**（不要为了聚合先 DTOH）；随后 **独立的** `VAggregate` kernel 在 device 上做归约，只把最终 `H×W` 拷回 host。
- `BM3Dv2` 在同一 `getFrame` 里连续 launch 这两个核，中间缓冲不暴露为 VS 帧。两段脚本 API（`BM3D` + `VAggregate`）若中间必须出 VS 帧，则不可避免 DTOH 胖帧——GPU 树上 **推荐糖接口 `BM3Dv2`**，两段 API 留给 CPU 和需要看中间量的调试。
- 第一轮不接 VS GPU VideoNode。

CUDA 特有：`.cu`、stream、pinned host、可参考 nlm-cuda / BM3DCUDA。
Vulkan 特有：构建期 `glslangValidator`/`glslc` 出 SPIR-V；计算队列；descriptor set 在创建期搭好；validation layer 仅 Debug。

CMake 默认 `NSS_ENABLE_CUDA=OFF`、`NSS_ENABLE_VULKAN=OFF`。`params.hpp` 从 Phase 0 起三边共用。

---

## 7. Host 行为（所有滤镜共用）

1. `fmParallel`。
2. `VSFilterDependency`：`radius==0` 用 `rpStrictSpatial`（若 API 允许）否则 `rpGeneral`；`radius>0` 对 `[n-R, n+R]` `rpGeneral`。
3. Workspace：`unordered_map<thread::id, Buffer>` + `shared_mutex`，对齐 64 字节。滤镜销毁时释放。
4. 只处理 `sigma[plane]!=0`（或 NLM 的 channels 掩码）的平面；其余 `copyFrame`。
5. 输出帧属性从输入 clip 复制。
6. `rclip`/`ref`：同分辨率、同格式、同帧数；匹配在 ref 上做，滤波在 clip 上做（BM3D Final 的 Wiener 同时读两者）。

---

## 8. 质量与测试

层次：

| 层 | 内容 | 何时 |
|---|---|---|
| 单元 | 小 SVD vs Eigen/LAPACK 残差；8×8 DCT 往返；SSD 与朴素循环 | Phase 0–1 |
| 核 | 合成 AWGN 小图（64×64 / 128×128）PSNR | 每个算法落地时 |
| 插件 | `vspipe` 对参考插件，固定种子噪声 | Phase 2 起 |
| 回归 | 禁止 GPU vs CPU bitwise；记录 PSNR 下限 | CI |

对拍对象：

| 本滤镜 | 参考 | 容忍 |
|---|---|---|
| `nss.NLM` | `nlm_ispc.NLMeans` 同参 | 优先接近；Highway 与 ISPC 允许 ULP 差，PSNR 差目标 &lt; 0.05 dB（合成图） |
| `nss.BM3D` | `bm3dcpu.BM3D` 同参、`radius=0` | 实现差（无 Kaiser、固定 group 8）允许；记录基线 PSNR |
| `nss.WNNM` | `wnnm.WNNM` 同参、`radius=0`、`residual=0` | SVD 路径不同，PSNR 目标差 &lt; 0.1 dB |
| `nss_cuda.NLM` | 本仓库 `nss.NLM` | PSNR，不比 bitwise |
| `nss_vk.NLM` | 本仓库 `nss.NLM`（次选 `nss_cuda.NLM`） | 同上 |
| `nss_cuda` / `nss_vk` BM3D | 本仓库 `nss.BM3D` | 同上 |

CI（Linux x86）：编译 + 单元测试。`vspipe` 对拍作为 `NSS_REF_TEST=ON` 手工/夜间，不强制无参考插件的机器。

---

## 9. 构建

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DNSS_ENABLE_CUDA=OFF \
  -DNSS_ENABLE_VULKAN=OFF
cmake --build build -j
```

依赖：

- CMake ≥ 3.24
- C++20 编译器（gcc 12+ / clang 16+）
- VapourSynth 头文件（pkg-config `vapoursynth`）
- Highway：FetchContent 钉 **git tag**（写入 `cmake/Highway.cmake`，升级单独 PR）
- CUDA 树：CUDA toolkit（Phase 5 才要求）
- Vulkan 树：Vulkan headers + loader + `glslc`/`glslangValidator`（Phase 6 才要求）
- 无 ISPC、无 MKL、无 Eigen 运行时依赖（Eigen 仅测试可选）

编译选项：`-O3 -ffast-math` 仅加在 Highway 核 TU；host 校验代码不用 fast-math。

安装：`libnss.so` → vapoursynth 插件目录（pkg-config `libdir/vapoursynth`）。

---

## 10. 阶段计划（施工顺序）

原则：每个阶段结束时仓库可编译、可运行对应滤镜或测试，不留空 stub 当完成。
**总顺序：CPU 金标准 → CUDA 把 GPU 算法定死 → Vulkan 做可移植平移。** 不要在 CPU NLM 之后就插 GPU，也不要 CUDA 与 Vulkan 两线同时探索 BM3D。

### Phase 0 — 空厂能加载

**目标：** `vspipe` 能 `core.nss.Version()` 或一个 identity 滤镜，证明 API4 + CMake + Highway 链接。

- CMake、FindVapourSynth、Fetch Highway
- `plugin.cpp`：`configPlugin` + `Version`
- `workspace` 骨架
- CPUID：无 AVX2 则加载滤镜时报错
- CI：configure + build

**完成定义：** `libnss.so` 被 autoload，`core.nss.Version()` 返回 git describe。

**不做：** 任何降噪。

### Phase 1 — `nss.NLM` CPU

**目标：** 单平面/多平面 f32 NLM，算法跟 vs-nlm-ispc（距离图 → 水平/垂直盒滤波 → 权重累加 → finish）。

- Highway 实现 distance / box / accum / finish
- `wmode` 先只做 0（Welsch）；其它模式随后
- `rclip`、`d/a/s/h`、`channels`
- 单元测试 + 小图 vs nlm-ispc（若本机有）

**完成定义：** 1080p `d=1,a=2,s=4` 能跑；合成噪声 PSNR 与参考同量级。

### Phase 2 — 共用 BM + `nss.BM3D`（radius=0）

**目标：** 搜索与聚合成为可复用 CPU 模块；BM3D Basic/Final 空间域可用。

- `match.cpp`：空间窗 SSD、无阈值纳入 group（与 WNNM/BM3DCUDA 空间策略一致）
- `dct8.cpp` + hard / Wiener
- `agg.cpp` 加权写回
- `ref` 行为与 BM3DCUDA 文档一致
- `group_size=8, block_size=8` 写死

**完成定义：** 对 `bm3dcpu` 同参 `radius=0` 有记录的 PSNR 基线；无崩溃、无 NaN。

### Phase 3 — 时间域 BM3D + WNNM

**目标：** 同一套 `CpuBm::match` 接 WNNM 小 SVD；`radius>0` 走独立 `nss.VAggregate`。

- predictive search（WNNM 语义）
- `svd_tiny.cpp` + 加权核范数 + 自适应聚合
- `residual` 开关
- `nss.VAggregate`（CPU）+ `nss.BM3Dv2` 糖（内部 invoke 两步）
- SVD 单元测试 vs 朴素实现

**完成定义：** `nss.WNNM` 与参考插件合成图 PSNR 差在 §8 阈值内；`radius>0` 走本插件 `VAggregate`，不依赖 HOVE。

### Phase 4 — CPU 加固（短）

只收尾，不在这里开新算法。

- NLM `wmode` 其余模式（若要 drop-in nlm-ispc）
- README 使用示例
- 记录 CPU 性能数量级（不是门禁）

**完成定义：** CPU 三条滤镜可给脚本用；Phase 5 有对拍基准。

### Phase 5 — CUDA NLM

**目标：** 第一条封闭 GPU 流水线。算法已在 Phase 1 定死，这里只换执行器。

- `libnss_cuda.so` + `nss_cuda.NLM`
- pinned + `num_streams=2`，参考 nlm-cuda
- 对拍 `nss.NLM` PSNR

**不做：** BM3D、Vulkan、GPU WNNM。

### Phase 6 — CUDA BM3D + CUDA VAggregate

**目标：** GPU NSS 主路径。协同滤波与聚合是 **两个 device kernel**；性能来自「中间量不回 host」，不是把两步熔成一个核。

- `nss_cuda.BM3D`：`radius=0` 直接出最终帧；`radius>0` 出 device 中间量（糖路径不暴露胖 VS 帧）
- `nss_cuda.VAggregate`：独立聚合核（这是相对 CPU `VAggregate` 的主要加速点）
- `nss_cuda.BM3Dv2`：同一次 `getFrame` 里 launch BM3D 核 → VAggregate 核 → DTOH 最终帧
- 对拍 `nss.BM3D` + `nss.VAggregate`

**不做：** CUDA WNNM；不做 GPU 中间量 → CPU `VAggregate`。

### Phase 7 — Vulkan NLM

**目标：** 把 Phase 5 已验证的 NLM GPU 算法译成计算着色器，证明 Vulkan 宿主（队列、描述符、SPIR-V 构建）能跑 NSS。

- `libnss_vk.so` + `nss_vk.NLM`
- 构建期编译 shader
- 对拍 `nss.NLM`（有 CUDA 机器也可与 `nss_cuda.NLM` 交叉）

**不做：** 在 Vulkan 上重设计 NLM；不接 VS GPU VideoNode。

### Phase 8 — Vulkan BM3D + VAggregate

**目标：** 移植 Phase 6：独立 BM 核 + 独立聚合核，不是第二份探索。

- `nss_vk.BM3D` / `nss_vk.VAggregate` / `nss_vk.BM3Dv2` 糖
- 对拍 `nss.BM3D` + `nss.VAggregate`

### 以后（不排进当前计划）

- CUDA / Vulkan WNNM（batched 小 SVD，各写各的）
- VS GPU VideoNode（R80+）绑定到 `nss_vk`（届时仍禁止和 CUDA 混）
- 第二波 NSS：MCWNNM、TWSC（优先，与现有 WNNM 共享搜块/SVD），然后 NLH、LSSC、NCSR（§14）
- `chroma` CBM3D、整数格式、RGB2OPP、Windows wheel
- NLM `wmode≠0`（Phase 4 若未做）

---

## 11. 阶段内的 PR 切分

每个 PR 保持可编译。依赖按编号。

| PR | 标题 | 阶段 | 依赖 |
|---|---|---|---|
| P0.1 | 仓库骨架：CMake、目录、`docs/plan.md` | 0 | — |
| P0.2 | API4 插件入口 + `nss.Version` + AVX2 门闩 | 0 | P0.1 |
| P0.3 | Highway FetchContent + 空 `HWY_DYNAMIC_DISPATCH` 烟测 | 0 | P0.2 |
| P1.1 | NLM host 参数校验与 workspace | 1 | P0.3 |
| P1.2 | NLM Highway 核（luma f32） | 1 | P1.1 |
| P1.3 | NLM 多平面 / channels / rclip | 1 | P1.2 |
| P1.4 | NLM 测试与参考对拍脚本 | 1 | P1.3 |
| P2.1 | `SearchConfig` + 空间 block match + SSD | 2 | P0.3 |
| P2.2 | 8×8 DCT + hard threshold + aggregate | 2 | P2.1 |
| P2.3 | `nss.BM3D` host（Basic，radius=0） | 2 | P2.2 |
| P2.4 | Wiener Final（`ref`） | 2 | P2.3 |
| P3.1 | 小 SVD 库 + 单测 | 3 | P0.3 |
| P3.2 | WNNM shrink 接到 CpuBm | 3 | P2.1, P3.1 |
| P3.3 | `nss.WNNM` host | 3 | P3.2 |
| P3.4 | `nss.VAggregate` + predictive search；`BM3Dv2` 糖 | 3 | P2.4, P3.3 |
| P4.* | CPU README 与 wmode | 4 | Phase 3 |
| P5.1 | CUDA 插件骨架 + stream 池 + 空 Version | 5 | Phase 3 稳定 |
| P5.2 | `nss_cuda.NLM` | 5 | P5.1, Phase 1 |
| P6.1 | `nss_cuda.BM3D` 空间域 | 6 | P5.2, Phase 2 |
| P6.2 | `nss_cuda.VAggregate` 独立核 | 6 | P6.1 |
| P6.3 | `nss_cuda.BM3Dv2` 糖：两核 + 最终帧 DTOH | 6 | P6.2, Phase 3 |
| P7.1 | Vulkan 插件骨架 + SPIR-V 构建 | 7 | P5.2 证明 GPU NLM 算法可用 |
| P7.2 | `nss_vk.NLM` | 7 | P7.1 |
| P8.1 | `nss_vk.BM3D` / `BM3Dv2` | 8 | P7.2, P6.2 |

---

## 12. 风险

| 风险 | 缓解 |
|---|---|
| Highway DCT 与 bm3dcpu 的 FFTW 尺度不一致 | 单测往返；sigma 映射可能要标定，文档写明 |
| 自研 SVD 与 MKL `sgesdd` 截断点不同导致 WNNM 观感差 | 单测残差；必要时 Jacobi 迭代上限与 ε 写成常量并与参考对拍 |
| NLM 盒滤波边界与 ispc 版 clamp 不一致 | 逐函数对照 vs-nlm-ispc 的 Horizontal/Vertical |
| 时间域搜索与 BM3DCUDA 不一致被用户当成 bug | 文档写死：跟 WNNM/V-BM3D，不跟 CUDA 加速搜索 |
| 把 VAggregate 熔进 BM3D kernel「看起来少一次 launch」 | 拆核；BM3Dv2 只做宿主层串联。bm3dcuda 上 GPU 聚合的提速来自少 DTOH，不是少 launch |
| FetchContent 无网 CI | 缓存 Highway tarball 或 submodule 作为备选（到 P0.3 再定一种） |
| 过早抽 CUDA/Vulkan 公共 GPU 层 | 等 NLM 在两套后端都落地后再看重复的 host 胶水，禁止为「统一」推迟第一套 GPU |
| Vulkan BM3D 无参考实现 | 只允许移植已在 CUDA 上对过拍的算法；禁止在 Phase 7 之前写 Vulkan BM3D |
| VS R80 GPU frame 诱惑 | Phase 5–8 全部走 CPU 帧进出；原生 GPU node 单列以后 |

---

## 13. 当前下一步（历史执行指针）

以下 P0.1/P0.2 是原始施工图中的历史执行指针；对应的 CMake 骨架、API4 入口和 `nss.Version` 已存在。当前 CPU 收尾工作见本节末的剩余执行计划。
Highway 烟测放在紧随其后的 P0.3，避免第一天就陷入 target 宏。

未决且 **不阻塞 P0** 的事项（需要时再改本文）：

- Highway 钉哪一个 upstream tag
- `nss.Version` 是独立滤镜还是只在插件描述字符串里
- 插件 id 是否改用个人/组织域名（现用 `com.nssfactory.nss` 占位）

---

## 14. 未实现算法与函数复用（第二波）

第一版 CPU 金标准已落地：`nss.NLM`（仅 `wmode=0`）、`nss.BM3D`/`BM3Dv2`/`VAggregate`、`nss.WNNM`。本节只排队，不改 D3/D5/D7/D8，也不冻 `block_size=8`。

参考（行为对拍，不 vendor Matlab）：

- [MCWNNM, ICCV 2017](https://arxiv.org/abs/1705.09912) / [csjunxu/MCWNNM-ICCV2017](https://github.com/csjunxu/MCWNNM-ICCV2017)
- [TWSC, ECCV 2018](https://arxiv.org/abs/1807.04364) / [csjunxu/TWSC-ECCV2018](https://github.com/csjunxu/TWSC-ECCV2018)
- NLH (Hou/Xu 2019, lifting Haar)、LSSC (Mairal 2009)、NCSR (Dong 2013) 仍无稳定 VS 参考，排在 MCWNNM/TWSC 之后

### 14.1 算法一句话

| 算法 | 输入 | 协同滤波 | 相对本厂的位置 |
|---|---|---|---|
| **MCWNNM** | RGB 联合（YUV444/RGBS） | 拼接 patch 成 `3p²×n`，行权 `W=min(σ)/σ_c`，ADMM；Z 步 = ClosedWNNM | WNNM 的多通道 + 加权数据项；无闭式解所以多 ADMM |
| **TWSC** | 灰度或 RGB 拼接 | 三权：通道/行噪声 `W1`、列噪声 `W2`、稀疏权 `Wsc`。组内 SVD 当字典，软阈值（`λ1=0`）或 ADMM | 同一套搜块；滤波从核范数换成稀疏编码 |
| **NLH** | 灰度（可后扩 RGB） | 块匹配 → 像素级 NSS → lifting Haar + 双 hard，再 Wiener | BM 骨架 + 新 Haar/像素匹配；`haar.cpp` 仍预留未建 |
| **LSSC** | 灰度 | 聚类（不是窗口 top-k）+ 联合稀疏 + 过完备字典 | 搜块语义不同；字典学习新核 |
| **NCSR** | 灰度 | 窗口 BM + 稀疏编码噪声（对非局部均值中心化）+ 迭代正则 | 搜块可复用；稀疏编码与 TWSC 近 |
| NLM `wmode≠0` | 整帧距离图 | 已有 distance/box/accum；换垂直权函数 | 不走 CpuBm |
| `chroma` CBM3D | YUV444 | Y 上匹配，UV 用同一坐标 | 匹配已有；pack 需多平面 |
| CUDA/Vulkan | 已验证算法 | 整条 device，禁止 CPU 搜 + GPU SVD | 无 CPU 核复用（D3） |

MCWNNM 模型：`min_X ‖W(Y−X)‖_F² + ‖X‖_{w,*}`。ADMM 拆成对角加权 X 步 + `svd_economy` + ClosedWNNM。ClosedWNNM 就是现有 `wnnm_shrink` 的奇异值更新：`σ̂ = (s + sqrt(s² − C))/2`（`ClosedWNNM.m`）。

TWSC 热路径（`λ1=0`，AWGN demo）：去均值 → `svd_economy` 得 `D` → `S = sqrt(max(s² − n σ_col², 0))` → `C = soft(DᵀY, σ_col² / S)` → `X = DC` → 按 `W2` 聚合。`λ1≠0` 才上 ADMM。

### 14.2 已有函数（直接复用）

| 已有 | 文件 | MCWNNM | TWSC | NLH | LSSC | NCSR | 备注 |
|---|---|---|---|---|---|---|---|
| `spatial_match` / `predictive_match` / `ssd_block` | `match.cpp` `ssd.cpp` | ✓ | ✓ | ✓ 块级 | ✗ 聚类 | ✓ | 窗口 top-k SSD；LSSC 另写 cluster |
| `pack_patch` / `unpack_patch` | `pack.cpp` | 单通道行 | 单通道行 | ✓ | ✓ | ✓ | RGB 拼接见 14.3 `pack_patch_nch` |
| `aggregate_add` / `aggregate_finish` | `agg.cpp` | ✓ 每通道 | ✓ | ✓ | ✓ | ✓ | 列权 `W2` 时 `w` 按 patch |
| `svd_economy` | `svd_tiny.cpp` | ✓ 每 ADMM 步 | ✓ 每组一次 | ✗ | 字典步 | 字典步 | `kSvdMaxM=256` 够 `3×8²=192`；`block=16` 的 RGB 要抬到 768 或拒绝 |
| `wnnm_shrink` / ClosedWNNM | `shrink.cpp` | Z 步抽出 SV 核 | ✗ | ✗ | ✗ | ✗ | 拆 `wnnm_sv_shrink(S, C)` 给 ADMM 复用，不要 fork 一份公式 |
| `residual` 去均值 | `shrink.cpp` | 常用 | 必做 | 可选 | 可选 | 可选 | 提到 `group_center` |
| `vaggregate_reduce` | `agg.cpp` | `radius>0` | 同 | 同 | 同 | 同 | 时间域协议不变 |
| `dct_1d` / `dct_lines` | `dct8.cpp` | ✗ | ✗ | 备选 | ✗ | ✗ | NLH 主路径是 Haar 不是 DCT |
| NLM distance/box/accum | `nlm.cpp` | ✗ | ✗ | ✗ | ✗ | ✗ | 整帧距离图，不进 BM 组 |
| host：`fmParallel`、workspace、validate、fat 布局 | `src/host/*` | ✓ | ✓ | ✓ | ✓ | ✓ | 新滤镜只换 Filter |

`kSvdMaxN=32` 对论文默认 `nlsp≈70` 不够。第二波要么 `group_size` 上限仍 32（工厂合同），要么单独抬 N；不要为 Matlab 默认改第一版 WNNM 上限。

### 14.3 未实现部分的共有新函数（只写一次）

MCWNNM 与 TWSC **同一作者、同一搜块、同一拼接、同一 ADMM 外壳**。优先抽这些，再写各自滤波：

| 新核 | 消费者 | 行为 |
|---|---|---|
| `pack_patch_nch` / `unpack_patch_nch` | MCWNNM、TWSC、CBM3D | 多平面 patch 拼成 `nch·p²` 列；匹配可只在 Y 或 RGB 拼接距离 |
| `channel_weight_diag` | MCWNNM、TWSC | `W_c = min(σ)/σ_c`（MCWNNM `NSig`；TWSC 行权同源） |
| `group_center` | MCWNNM、TWSC、WNNM residual | 行均值加减 |
| `wnnm_sv_shrink` | WNNM、MCWNNM ClosedWNNM | 现 `wnnm_shrink` 里的 SV 更新抽出来 |
| `admm_weighted_x` | MCWNNM、TWSC `λ1≠0` | `X = (W²Y + ρ/2 (Z − A/ρ)) / (W² + ρ/2)` 对角步 |
| `soft_threshold` | TWSC、NCSR、LSSC | `sign(B)·max(\|B\|−τ, 0)` |
| `iter_regularize` | TWSC、NCSR、MCWNNM 外环 | `x += δ(y − x)` |
| `noise_estimate_pca` | MCWNNM/TWSC 实噪声 | Liu PCA（`NoiseEstimation.m`）；第一版可只吃用户 `sigma[]` |
| `haar_2d` / `haar_group` | NLH（BM3D haar 预留） | lifting Haar；与 DCT 并列，不替换 `dct8` |
| `pixel_match` | 仅 NLH | 组内行/像素匹配，现有 `spatial_match` 不够 |

不要抽的：

- K-SVD / 过完备字典：只有 LSSC/NCSR，且无 VS 参考
- 像素级 NSS：只有 NLH
- NLM `wmode` 权函数：只留在 `nlm.cpp`
- GPU 核：D3，CPU 搜块不进 device SVD

### 14.4 建议落地顺序（仍属「以后」）

1. 抽出 `wnnm_sv_shrink` + `group_center`（WNNM 行为不变）
2. `pack_patch_nch` + 可选拼接距离的 match
3. `nss.MCWNNM`：host 抄 WNNM，Filter = ADMM(`admm_weighted_x` + SVD + `wnnm_sv_shrink`)；RGB/YUV444；`sigma` 三通道
4. `nss.TWSC`：同 host/pack；`λ1=0` 先用 `soft_threshold`；ADMM 后做
5. NLH：补 `haar.cpp` + `pixel_match`（与 1–4 的新核几乎不交，可后做或与 3 分文件并行）
6. NCSR：等 TWSC 的 `soft_threshold` / 组内 PCA 字典稳定
7. LSSC：最后。聚类 + 过完备字典 + 联合稀疏（ℓ_{1,2}），不复用窗口 top-k

GPU：第二波算法默认 CPU-only（与现 WNNM 相同）。CUDA/Vulkan 仍只跟已对拍的 NLM/BM3D。

### 14.5 NLH / LSSC / NCSR：不要当成一轮大并发

三者都是 NSS，但 **Filter 核几乎无交集**。能共用的已经在第一版 BM host 里；还没写的新核各吃各的。开三个 agent 同时写滤镜，只会在 `match.cpp` / `pack.cpp` / `cpu_api.hpp` / `plugin.cpp` 上撞车，并复制三份「自己的稀疏/变换」。

流水线（只写和本厂有关的步）：

```text
NLH:  spatial_match → pack → pixel_match(行) → lifting Haar → 双 hard
       → agg →（stage2）Haar(noisy, basic) → Wiener
NCSR: spatial_match → pack → group_center → PCA/字典 → 逐列稀疏
       → 码域非局部均值 → 收缩编码噪声 → iter_regularize → agg
LSSC: 抽全体 patch → cluster（Si=Sj）→ 过完备 D（K-SVD/ODL）
       → 组 ℓ_{1,2} 联合稀疏（OMP/LARS，不是软阈值）→ agg
```

| 核 | NLH | NCSR | LSSC | 已有？ |
|---|---|---|---|---|
| `spatial_match` / pack / agg / host | ✓ | ✓ | pack/agg/host | 已有。LSSC **不用** 窗口 top-k |
| `pixel_match` + `haar_*` | ✓ | ✗ | ✗ | 无。只 NLH |
| `soft_threshold` + 组内 SVD 当 D | ✗ | ✓（近 TWSC） | ✗ | 无。等 TWSC |
| `iter_regularize` | 外环 K≈2，语义不同 | ✓ | 可选 | 无。一行，NCSR/TWSC 用 |
| cluster + 过完备 D + OMP/ℓ_{1,2} | ✗ | 字典可 PCA 不是 K-SVD | ✓ | 无。只 LSSC 值得新写 |
| BM3D `dct_lines` / `wnnm_shrink` | Haar 不是 DCT | 不是核范数 | 不是核范数 | 不复用进这三家 Filter |

因此：

- **不能**「NLH+LSSC+NCSR 一起做」。公共设计已经收口（CpuBm host）；公共实现没有待写的大块。
- **NCSR 依赖 TWSC 的稀疏原语**，不要和 TWSC 并行写两套 `soft_threshold` / PCA-D。
- **NLH 文件面独立**（新 `haar.cpp`、`pixel_match`、`filter_nlh.cpp`），与 MCWNNM/TWSC 可以分文件并行，但 **不要** 改 `spatial_match` 的 top-k 合同。
- **LSSC 单独最后**。无 VS 参考；SPAMS/OMP/字典学习比另外两家都大，且聚类语义与现有 match 冲突。

多 agent 只允许在 **共享核已合入、文件不重叠** 之后：

| 阶段 | 谁 | 写哪些 | 禁写 |
|---|---|---|---|
| S0 一人 | 抽接口 / 实现已有复用点 | `group_center` 若尚未抽；`cpu_api.hpp` 声明 | 三个滤镜 host |
| S1 至多两路 | A: NLH（haar + pixel_match + filter_nlh）B: MCWNNM 或 TWSC | 各自新文件 + 自己的 `filter_*.cpp` | 对方的核；`match.cpp` 控制流 |
| S2 | NCSR | `filter_ncsr.cpp`，调用 TWSC 已落地的 soft/PCA | 新字典求解器 |
| S3 | LSSC | `cluster.cpp` `omp.cpp` `filter_lssc.cpp` | 改 `spatial_match` 当聚类 |

禁止：三个滤镜 agent 同时改 `plugin.cpp` / `params.hpp` / `match.cpp`；禁止为「公用」先抽一层抽象稀疏框架再填三家。先落地一家（NLH 或 TWSC 稀疏），第二家调用，第三家 LSSC 再决定要不要共享 OMP。

## 15. CPU 综合优化剩余执行计划

本节是 2026-08-31 之后的执行基线，只覆盖 CPU 综合优化。CUDA/Vulkan 原始阶段仍是历史规划，不影响本轮完成定义。外部参数、默认值、输出布局、迭代次数和单组 CPU API 必须保持不变；batch 只能作为内部能力。

### 15.1 Workstream 与依赖

| Workstream | 依赖 | 当前状态 | 完成 gate |
|---|---|---|---|
| W0 正确性与观测 | corrected SVD | 部分完成 | 独立 corrected-baseline checkpoint；JSON 固定 revision/compiler/CPU/参数/尺寸/seed/thread/warmup/wall-time 字段 |
| W1 matcher/batch | W0 | 实现完成 | scalar-vs-batch differential；稳定 tie/NaN；短尾与有界 ordered commit；不能丢 group |
| W2 BM3D | W1 | 部分完成 | Basic/Final、ref、radius 0/1、fat intermediate、非默认参数；C4 至少 `1.25x` |
| W3 WNNM/SVD | W1 | 部分完成 | 重构/正交/奇异值阈值、失败 group 行为和参考 PSNR；C4 至少 `1.15x` |
| W4 NLM | W0 | 部分完成 | Gray/UV/YUV/RGB、`d/a/s` 边界、stride/subsampling；C4 1080p Gray `d=1` 至少 `1.30x` |
| W5 TWSC → MCWNNM → NCSR → NLH | W1、各自前项 | 部分完成 | 每个滤镜独立 checkpoint、scalar differential、plugin smoke、1/2-thread hash；TWSC/MCWNNM/NCSR `>=1.15x`，NLH `>=1.30x` |
| W6 LSSC prepared | W0、W1 | 部分完成 | frame-local D/Dᵀ/Lipschitz、prefix members、workspace-only cluster/dict/OMP、有限值/PSNR；C4 `>=1.15x` |
| W7 正式 campaign | W2–W6 | 未完成 | C4 runner 生成完整 artifact，所有对应 gate 通过，文档矩阵与 artifact 一致 |

W5 的 batch 接入顺序固定为 **TWSC → MCWNNM → NCSR → NLH**；全局 group 顺序固定为 **BM3D → WNNM → TWSC → MCWNNM → NCSR → NLH → LSSC**。任何 lane 在总工作负载回退超过 2% 或质量超出阈值时停止集成并回到最近的正确性 checkpoint。

### 15.2 实现约束与 stop rule

- matcher 的排序键是距离、self-first、时间、坐标、原始候选 ordinal；有效距离优先于 NaN/Inf。`block=8/group=8` 的快速路径遇到非有限输入必须回退稳定通用路径。
- batch wrapper 可以按 `(m,k,channels,algorithm,basic,residual)` 分桶，但当前“分桶后逐项调用 kernel”不计作跨 group lane SIMD；在没有 workload 级证据前不宣称 lane 加速。
- NLM 继续使用 stripe-local/halo 和每线程最多 1 MiB scratch；mixed plane stride 必须走 stride-aware 路径，不能把 plane 1 的 stride 传给所有 plane。若新 lane 达不到 C4 gate，保留正确性版本并撤回实验 lane，NLM 未过 gate 时不得宣布 CPU 综合计划完成。
- LSSC prepared context 只在当前帧存活，不缓存输入相关字典；cluster、dictionary、OMP、reconstruct 的热路径临时存储必须来自调用者 workspace，不能用隐式 heap allocation。
- PMU 只作诊断。只有 guest 上实际非零硬件计数和成功 `perf record` 才能写入证据；`perf list`、CPU 文档或单次 top-down 百分比不能作为合并条件。

### 15.3 C4 正式 runner

入口固定为：

```text
tests/run_c4_cpu_gate.sh \
  --project <project> \
  --zone <zone> \
  --vs-venv-tar <path>
```

runner 要求调用者通过 `NSS_C4_BASELINE_REF` 或 `NSS_C4_BASELINE_DIR` 显式提供唯一的 corrected baseline；不再默认使用可能包含错误 SVD 的 `HEAD`。自动 runner 仍可做 zone、quota、machine type 和唯一实例名的 fail-closed 检查，但 Policy Troubleshooter API 不是手动 SSH campaign 的前置条件：用户已确认创建/删除权限时，直接创建唯一实例、SSH 执行、拉回 artifact，再手动删除并确认不存在。macOS 打包设置 `COPYFILE_DISABLE=1`，防止 AppleDouble `._*.cpp` 在 Linux guest 被 CMake glob 当成源码。

guest 使用传入的预构建 VapourSynth venv tar，不在线安装或替换运行时。baseline/candidate 源码和构建配置分别上传，在独立进程中交错运行；测试进程固定到 CPU 0，并记录初始化、独立 warmup、不同 frame number、单/多线程元数据和三种内容形状。当前 WNNM、TWSC、MCWNNM、NCSR、NLM、NLH、LSSC 已分别生成正式 C4 artifact，不能互相替代；BM3D 仍须单独达到 `1.25x` 并补齐合法配置矩阵，才能生成 passing campaign。

退出时的 `trap` 只删除本次创建的实例并轮询确认删除；Spot admission、SSH、guest build、artifact 或删除任一步失败，都报告精确状态而不创建 fallback 资源。`tests/c4_guest_gate.sh` 只负责 guest build/JSON 标准库解析和 paired 统计；历史 `310adb0` 的旧文本 benchmark 输出仅作兼容解析，最终 artifact 仍使用当前 schema。

### 15.4 2026-08-31 全算法 Topdown 基线

统一条件为 1920x1080、固定 seed、单 VapourSynth 进程、单线程固定 CPU 0，sibling CPU 1 空闲；guest 是 Xeon Platinum 8581C 的 1 物理核/2 SMT、2 MiB L2。每个算法均采集 Topdown L1/L2/L3 和独立 `cycles:u perf record`。L3 细分事件仅约 2.94% running，因此只作方向诊断，不作为性能 gate。

| 算法 | ms/frame | Backend | Core | Memory | `perf record` 主要热点 |
|---|---:|---:|---:|---:|---|
| NLM | 228.0 | 63.6% | 45.0% | 19.5% | VerticalWelsch 40.5%，Horizontal 26.3% |
| BM3D | 82.5 | 35.5% | 26.0% | 10.3% | SpatialMatch8 58.2%，Bm3dFilter8 19.4% |
| WNNM | 247.4 | 38.1% | 25.3% | 12.9% | JacobiSvd8 27.7%，ApplyHouseholder 20.3%，matcher 19.7% |
| TWSC | 529.0 | 37.4% | 25.0% | 12.9% | Jacobi 25.5%，matcher 19.0%，Householder 18.8% |
| NCSR | 550.0 | 40.8% | 26.0% | 14.4% | Jacobi 24.7%，Householder 18.2%，matcher 17.9% |
| LSSC | 533.0 | 35.0% | 27.6% | 8.7% | GemmNN 44.9%，reconstruct 20.8%，SsdVec 12.0% |
| NLH | 9048.0 | 23.9% | 17.7% | 5.9% | SpatialMatch 31.5%，Haar 20.9%，SsdBlock 17.6%，PixelMatch 14.5% |
| MCWNNM | 5855.0 | 39.3% | 24.9% | 14.4% | Householder 29.8%，Jacobi 26.9%，DotN 6.9% |

完整 artifact 位于 `artifacts/c4/topdown-20260831-current/`：24 份 Topdown、8 份非空 `perf.data`、8 份 symbol report，目录归档 SHA-256 为 `3f48665408b40df41755b0a8c8a4b36c69cf04851ea4d8ea9c527f40b2a9d979`，guest 源码包 SHA-256 为 `569980db0d746406b7ab09e1061dec5028f9d8712b016adad3c599fe0a952654`。

2026-09-01 使用当前优化工作树、相同 C4/1920x1080/seed/affinity/frame-count 协议重跑：24 份 Topdown、8 份非空 `perf.data`、8 份 symbol report 全部完整，`perf record` 均为零 lost samples。相对上述 corrected profile，NLM/BM3D/WNNM/TWSC/NCSR/LSSC/NLH/MCWNNM 分别为 `1.2140x/1.1007x/1.4280x/1.3955x/1.3283x/1.3068x/1.3980x/1.5724x`；相对最初历史 session 的四舍五入耗时则有 7/8 变快，NLH 仍为 `0.8243x`。最初 VM 的 raw PMU 文件已随删除消失，因此该列明确标为历史近似；完整边界、Topdown 变化和当前热点见 `artifacts/c4/topdown-20260901-optimized/comparison.md`。当前 artifact 压缩包 SHA-256 为 `23f232bc99024afd7c8510f82420e14830cba12ad5e4d36e4a01b39c09433b50`。

第一次 Highway 实验把 matcher candidates 映射为 SIMD lanes，插件级三组 A/B 的中位数比值分别为 BM3D `0.3856x`、WNNM `0.6446x`、TWSC `0.6601x`、NCSR `0.6703x`、NLH `0.9618x`。该实验全面回退，已经精确撤回；`SpatialMatch8` 不再是近期重写目标。当前最高共享收益方向改为真正的跨 group `n=8` SVD/Householder Highway batch，然后分别处理 NLM pass fusion、NLH group=16 matcher/Haar、LSSC blocked GEMM。

BM3D 后续 first4 v7 复核使用与 `SpatialMatch8` 相同的四 FMA 累加器和 reduction tree，并验证所有可实现调度的最终 ordinal-stable top-8。纯 first4 的最强零开销 baseline ceiling 为 `1.216935x`；将 tail-DC 的 tail load/reduction、向下误差保护和分支也假设为零成本，最强可实现 ceiling 仍只有 `1.242308x`。因此不再投入 first4、seed warmup、center-order 或 tail projection 原型；BM3D 只评估能显著改变完整 matcher 或 `Bm3dFilter8` 吞吐的结构方案。

跨 reference AVX-512 原型把相邻两个 raster reference 及同偏移 candidate 放入一个 ZMM 的两个 8-lane 半区，保持原 FMA/reduction tree 和独立 stable top-8。它将总指令从约 `13.05B` 降到 `11.73B`，但 IPC 从 `2.51` 降到 `2.24`、cycles 从约 `5.20B` 升到 `5.24B`；7 对插件中位数仅 `0.993488750904x`，输出 hash 全同。该原型已完全撤回，不再扩展同类四-reference/ZMM 路径；证据见 `artifacts/c4/bm3d-refpair-20260901/summary.txt`。

`Bm3dFilter8` 的 final group-IDCT→num/den 直接聚合也已隔离测试：虽然删除了 64-vector `G` 写回/重读，row-major 写入重叠聚合缓冲使 7 对中位数降到 `0.940219322526x`。相对原路径 max abs `1.79e-7`、PSNR `155.24 dB`，质量兼容但性能明确失败，已完全撤回；证据见 `artifacts/c4/bm3d-invacc-20260901/summary.txt`。

matcher top-k 控制结构的 scalar max-heap 也已做隔离预检：OFF/ON C4 Release CTest 均为 14/14，但 7 对、每次 4,000 个默认 matcher 的 paired median 仅 `0.734050493338x`（中位 `6.380710 ms → 8.712793 ms`）。标量分支和 heap entry 搬移显著慢于当前 Highway mask/popcount/permutation 插入，因此未升级到完整插件 campaign，代码已完全撤回；证据见 `artifacts/c4/bm3d-heaptopk-20260901/summary.txt`。

### 15.5 完成定义

只有在 W0–W7 的实现、正确性、质量、性能和文档证据全部齐全，且 NLM C4 `>=1.30x`、各滤镜 gate 均通过时，才把 CPU 综合优化标记为完成。远端 `192.168.50.3` 的 5950X 只能用于快速开发回归；它不能替代 C4 正式 gate，也不能把本机 CTest 或 CI artifact 当成硬件性能证据。

### 15.6 2026-09-01 C4 PMU 快照基线

已将 C4 母机上的可复用依赖迁入 `/opt/nss-c4`，删除 `/tmp` 实验源码、构建树、性能数据、缓存和历史后创建全局快照 `nss-c4-pmu-env-20260901`（project `dsmvc-avx512-pmu`，storage location `us`，50 GB，状态 `READY`）。快照固定 Ubuntu 24.04、kernel `6.17.0-1022-gcp`、GCC 13.3.0、CMake 3.28.3、perf 6.17.13、Python 3.12.3、NumPy 2.5.2、VapourSynth R75 和 Highway 1.4.0 commit `2607d3b5b0113992fe84d3848859eae13b3b52c1`，并持久化 `perf_event_paranoid=-1`、`nmi_watchdog=0`。

独立恢复验证使用 `c4-highcpu-2` Spot 和 `STANDARD` PMU：9 个 1920x1080 Gray8 真实图片样本哈希全部通过，`cycles:u` 非零、`perf record` 非空且零 lost samples；网络隔离的全新 Release 构建为 14/14 CTest，并完成 `MAPPA.gray8` 的 BM3D 插件冒烟。恢复验证机及其 auto-delete boot disk、原母机及原 boot disk均已删除，只保留该 `READY` 快照。

后续 C4 实例必须通过 `/Users/owen/.codex/skills/nss-c4-pmu-spot/` 从该快照恢复，不允许在 snapshot restore 或 Spot admission 失败时退回公共镜像。每次实例恢复后仍须运行 `guest_verify.sh`，当前源码必须单独打包上传；真实图片 campaign 必须记录样本 basename/hash，不能和 fixed-synthetic 历史基线混作 A/B。该基础设施闭环不改变 15.5 的完成定义：BM3D 当前 `1.1007x < 1.25x`，W7 仍未完成。

### 15.7 2026-09-01 DCT codelet / NCSR FastExp 验证状态

本轮已完成并通过正式 C4 验证：`DctLines` 为 n=16/32/64 接入 genfft forward/inverse Highway codelet（8/16 lanes），n=2/4、n=8 和非 x86 fallback 合同保持不变；NCSR 两条权重路径统一为 Highway `FastExp` 动态分派。本机 Apple ARM Release CTest 与 C4 candidate Release CTest 均为 **14/14 passed**，C4 上已确认 16-lane codelet 实际编译执行。

正式 paired gate 只使用 `03144bc` 作为本轮 current boundary，不再把启用不同 Highway target 的 `ed97d8b` 混作 broad baseline。历史 NCSR SVD gate 的约 `1.3075x` 继续由 `artifacts/c4/svd-batch-20260831/ncsr-svd-paired.tsv` 独立保存；它不计入本轮 pass/fail。第一次 attempt 因 baseline 选择错误和统一 `120 dB` 质量阈值不适用于约 `1e-7` 的 DCT 重结合误差而作废，保留在 `artifacts/c4/dct-ncsr-round-20260901/attempt1/` 仅作失败审计。

最终 `nssfactory.c4.cpu-gate.v2` artifact 为 `artifacts/c4/dct-ncsr-round-20260901/final/attempt2/c4.json`：

- BM3D 12 个 block/group 配置相对 `03144bc` 最低 `1.015698x`，受影响配置中位数 `1.364205x`；默认 BM3D 输出 bit-exact。
- NCSR 1920x1080、7 对中位数为 `0.993227x`，满足本轮不回退超过 2% 的增量门槛；历史 `1.3075x` broad gate 不重算、不混算。
- LSSC block=8/16 最低 `1.005299x`；默认 LSSC 输出 bit-exact。
- candidate BM3D/NCSR 重复运行 hash 一致。近似配置全部有限，最低 PSNR `114.2066 dB`、最大绝对误差 `7.29263e-5`，通过统一 `PSNR >= 110 dB && max_abs <= 1e-4` 门槛。
- candidate BM3D 与 NCSR 的 `cycles:u perf record` 分别为 100176 和 79332 bytes，两个 report 均为 `Total Lost Samples: 0`；恢复环境 `guest_verify.sh` 再次通过。

最终归档 `artifacts/c4/dct-ncsr-round-20260901/final/dct-ncsr-round-20260901-final.tar.gz` 的 SHA-256 为 `242a8604a2d26a430ad4bffaf6c48676f8ab849104ab1f61cf7ebd52818ae225`，下载后外层哈希与 11 个内部文件哈希全部复验通过。Spot 实例 `nss-c4-dct-ncsr-20260901-a` 及 auto-delete boot disk 已删除并分别确认不存在。

因此 **DCT codelet / NCSR FastExp 这一轮优化完成**，但 15.5 的整个 CPU campaign 完成定义不变：本轮 codelet 不触及默认 BM3D n=8 热路径，默认 BM3D 仍未达到 `1.25x`，W2/W7 和 CPU 综合计划不能标记为完成。


## 2026-09-05 优化复盘与维护补充

本节补充当前状态，保留前面的历史设计。b4/b12/b16、最新 b16 output sink、
VAggregate 整合、失败候选及跨算法适用边界见 [优化复盘](optimization-review-20260905.md)。

本轮保留 BM3D/NLH prepared commit、NLH 有界 scratch 复用和失效参数清理、
b8 未使用转置函数删除；BM3D scratch 复用试验未保留。C4 最终 15/15 CTest、
89 个插件对拍配置通过；六项七对性能复测均在 2% 退化门槛内，但 r1 约慢 1.58%，
属于明确的维护取舍，不能宣传为所有配置提速。证据与资源清理状态见
[本轮报告](../artifacts/c4/maintenance-20260905-a1/summary.md)。

下一轮优先建立真实距离流的 top-k replay、检查 b8 Wiener 活跃区间；
b12 输出布局可借鉴 b16 sink，但继续保留独立 TU、归约次序与整链 gate。


## 2026-09-05 DCT 布局与 b8/g8 实验补充

本节追加本轮状态，保留前面的历史设计与综合 CPU 目标。详细证据及候选表见
[布局实验报告](optimization-layout-20260905.md)，实验入口见
[kernel lab](bm-kernel-lab.md)。

- b12 cross-patch 九图真实 Basic→Final 为 1.015621×，但 g2 Basic 控制项
  在十五对后仍无法确认满足 1% 回退界限，保留 lab。8+4 sink 未通过整链筛选。
- b8 通用双 patch 及行转置内联均在普通插件 b4/g1 对拍失败；双 patch 的
  一次隔离接入修订也失败。b4 两种矩阵布局在 AVX2/AVX-512 微核对拍失败。
- b8/g8 TopK 表与 paired partial-sum 未获得整链收益；FFTW 转置内联
  九图整链为 1.054446×，但 g4 Basic 超过 1% 回退门槛，独立 TU 修订未解决。
- PMU 已区分 Basic/Wiener，未据此扩大成未经证据支持的 G/R/group-axis 重写。
- 通用 lab、方向计时、真实 patch corpus、TopK 分类 replay、保护页与 stride
  检查已实现；重复 A/B、数值对拍和 PMU 采集集中到 `tests/c4_*` 工具。
  新实验复用既有 inline 转置，实验默认关闭；公开参数、生产派发和浮点编译选项保持不变。

本轮没有接入新的生产提速候选。后续若继续 b8 布局，应先建立能隔离编译器
代码生成变化的浮点兼容边界，并保留跨 block 的冻结插件对拍。共享 matcher
目前仍保持原实现；多通道匹配、SVD、Haar 和 NLM 不扩大为独立重写。
最终维护回归、性能验收和临时资源清理见上述报告的最终验收部分；本节不将
历史 W2/W7 或整个 CPU 综合优化计划标记为完成。

维护控制矩阵的 NCSR 项在十五对后仍无法确认 1% 回退界限，因此运行时转置
去重也已撤回，不缩小预先冻结的矩阵来宣称通过。最终仅交付实验/测试工具
整合及测试覆盖，普通生产代码恢复原样，并以同 ISA 插件二进制一致性和重新
回归验证封闭；详见布局报告的最终交付部分。

最终工具版已完成 C4 验证：三种派发的普通插件与对应冻结基线逐字节一致，
各 15/15 CTest、197 项插件对拍通过；lab 17/17 CTest、两 ISA 校验、
rolling 和 VAggregate 回归通过。旧 b16/SSD lab 的 AVX2 不支持边界继续
显式报告为跳过，不伪造该路径的微核计时。最终证据和资源清理记录见
`artifacts/c4/layout-20260905-a1/summary.md`。


## 2026-09-05 BM3D correctness and dual-region C4 campaign

B0 d572c52 freezes the current dirty runtime and lab. P0 4154cdf corrects fixed
reference matching, real frame bounds, target aggregation, zero-sigma identity,
rolling dependencies, and effective BM3D sigma. Both regional C4 hosts passed
16/16 CTest plus independent public sigma, time and concurrency oracles.
P1/P2 evaluation and final default-off verification are recorded in
bm3d-correctness-campaign.md. No optimization candidate passed the full gate.

## 2026-09-06 BM3D campaign completion

The final P runtime includes the later overflow and Wiener-alias repairs, plus
exact unit-weight VAggregate identity on AVX2. Dynamic, AVX2 and AVX3 plugins
are byte-identical to independently assembled pure-correctness references;
each passed 16/16 CTest and 230/230 plugin comparisons on YUL. TLV independently
verified dynamic with the same binary hash. Both lab builds passed 18/18 CTest,
and retained rolling and VAggregate regressions passed.

The user's revised budget targets about 40 seconds per complete A/B bench group,
including calibration and warm-up. Frame counts are calibrated instead of using
thousands of fixed requests. Fewer than seven pairs, incomplete random groups,
and earlier cached-only ring timings cannot pass the admission gate. Completed
historical evidence is preserved. Every optimization remains default-off.

Migration covered 1,080 synthetic-motion/real-image cases; memory, allocation,
4K, matching replay and PMU diagnostics are archived. Natural-video acceptance
is unverified. The earlier B0-to-F temporal correctness-cost diagnostic shows
roughly 30%-35% more runtime, and is not an optimization claim. Full outcomes,
source/artifact hashes, residual proof boundaries, and exact two-VM cleanup
records are linked from the campaign report. This does not complete or change
the historical W2/W7 or overall CPU performance targets.

## 2026-09-06 Bounded per-configuration selection

Under the user's revised >1.02x acceptance rule, SortedTopK, bounded BM3D reuse,
and rolling ring/direct-scratch extraction form the selected default mask 2305.
Cached-worst remains an accepted alternative for its qualifying configurations;
lazy raster and homogeneous dispatch did not reach 1.02x in the bounded matrix.
Three C4 Spot hosts ran independent lanes with approximately 30-60 second group
budgets. All candidate numerical matrices and the combined dynamic/AVX2 checks
passed. Measured configuration results, single-pair limitations, source hashes,
raw audits and cleanup records are in [the selection report](c4-selection-20260906.md).
