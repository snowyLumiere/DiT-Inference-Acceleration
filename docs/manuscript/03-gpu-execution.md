# 第3章 GPU 执行与性能约束

把 GPU 简化为一个峰值 TFLOPS 数字，会掩盖大部分工程瓶颈。同一 DiT 层既有大 GEMM，也有归约、重排、短 Kernel 和数据拷贝；它们对 SM、存储层次和调度的需求截然不同。定位性能问题时，先区分受限于算力、带宽还是调度空档，再决定是增加并行度还是减少数据搬运。

## 从线程到 Tile

CUDA Kernel 把工作划分成 thread blocks；block 内线程以 Warp 为执行单位。一个 SM 能同时驻留多少 block/Warp 受线程数、寄存器、shared memory 及架构上限共同限制。Occupancy 反映活跃 Warp 相对架构上限的比例；更多 Warp 有助于掩盖访存等待，但也可能因寄存器压力限制单个 block 的 tile 规模。

GEMM 和 Attention Kernel 通常按 Tile 消费张量。假设输出矩阵为 $[M,N]$、输出 Tile 为 $[B_M,B_N]$，未考虑 split-K 时输出任务块数为：

$$\mathrm{tiles}_M=\left\lceil\frac{M}{B_M}\right\rceil,\qquad
\mathrm{tiles}_N=\left\lceil\frac{N}{B_N}\right\rceil,\qquad
\mathrm{output\_tiles}=\mathrm{tiles}_M\times\mathrm{tiles}_N$$

当 $M$ 很小，即使 $K$ 和 $N$ 很大，输出 Tile 数也可能不足以铺满整卡 SM。相反，增大 Tile 能提高片上数据复用，却会消耗更多寄存器和 shared memory；在长视频序列上高效的 Tile 未必适合短 chunk。分析时还需兼顾末尾不足一个 Tile 的边界浪费与 Wave 量化效应。

Tensor Core 执行的是满足特定指令与布局约束的矩阵运算。FP8 张量必须由匹配的 Kernel 消费才能走 FP8 Tensor Core；若 Shape、对齐或架构不匹配，底层可能回退到高精度路径。核对性能规格时，需区分 Dense 峰值与 2:4 结构化稀疏峰值，常规稠密算子以非稀疏峰值为参照。

## 存储层次与布局

权重和激活主要驻留设备显存（HBM/GDDR），数据经 L2/L1 cache 进入 SM，Kernel 可利用 shared memory 在 block 内复用数据，再送入寄存器做矩阵乘。L2 命中率直接决定落到显存颗粒上的真实流量；理论最小读写字节数给出下界，实际带宽消耗以硬件计数器为准。

例如 Q 的逻辑 Shape 同为 `[B,H,N,d]`，连续的 $d$ 维和连续的 $N$ 维对应不同的物理地址步幅（stride）。`transpose` 通常只创建带新 stride 的 view；若下游算子要求连续布局，隐式或显式的 `contiguous()` 就会触发真实内存拷贝。为了省一次转置而让主计算 Kernel 以低效 stride 访存，往往得不偿失，因此布局选择需要把生产者、转换和消费者放在同一段区间内衡量。

Coalesced access 关注同一个 Warp 的访存地址能否合并为尽量少的内存事务。逐 token 随机 gather、非连续 head 排列和细碎拷贝都会拉低有效带宽；按整块组织的稀疏模式更容易保持连续读取。执行和访存细节可参见[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

## Roofline 提供约束，不直接诊断整模型

设一个算子包含 $F$ 次浮点操作、实际穿过目标存储层的字节数为 $B$，硬件在对应精度下的有效计算吞吐为 $P$、有效带宽为 $\mathrm{BW}$，则理想重叠下的耗时下界为：

$$I=\frac{F}{B}\quad(\mathrm{FLOP/byte}),\qquad
T_{\mathrm{compute}}\ge\frac{F}{P},\qquad
T_{\mathrm{memory}}\ge\frac{B}{\mathrm{BW}}$$
$$T_{\mathrm{kernel}}\ge\max(T_{\mathrm{compute}},T_{\mathrm{memory}})$$

这个下界不包含 Kernel Launch、依赖等待、Tile 尾部浪费和非矩阵指令瓶颈。DiT 主干不总是 Compute-bound，AR Decode 也不总是 Memory-bound：batch、query/KV 长度、dtype 和布局都会改变算术强度。

以矩阵乘 `X[M,K] × W[K,N]` 为例，乘加按 2 FLOPs 计，输入输出每元素占 $b$ bytes 且理想情况下只读写一次：

$$F=2MKN,\qquad
B_{\min}=b(MK+KN+MN),\qquad
I_{\mathrm{ideal}}=\frac{2MKN}{b(MK+KN+MN)}$$

$M=1$ 时，读取权重 $W$ 的字节无法在多个 token 间摊销；$M$ 增大后权重复用率随之提升。连续 AR-chunk 的单块 token 数 $M$ 往往在千余至数千量级，其线性投影完全属于矩阵乘几何，而不是单 token Decode 的 GEMV。矩阵乘分块与带宽背景见 [NVIDIA GEMM 指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

## Kernel Launch 与依赖

CPU 可以异步向 GPU 队列提交工作。短 Kernel 密集串联时，Host 发射、Python 分支或隐式同步容易让 GPU 出现断流等待；而当单个大 Kernel 已经吃满设备时，单纯削减 launch 次数带来的收益就很有限。CUDA Graph 主要省去重复发射的 Host 开销，算子融合则重在省去中间结果的显存读写。

多 Stream 允许表达无依赖的并发工作，但不会凭空增加 SM、显存带宽或 DMA 引擎。如果两个大 Kernel 争抢同一硬件资源，并发执行反而可能拉长各自耗时。只有结合时间线（trace）观察实际重叠区间与关键路径变化，才能确认多流并发是否真正奏效。

## 架构差异如何进入实验

Hopper 与不同 Blackwell 产品的算子支持路径存在显著差异。面向 SM90（如 H100/H800）调优的 Kernel 不会自动适配 SM120（如 RTX 5090），同一架构代号下的不同芯片在共享内存配额、互联带宽和可用指令上也有区别。从“编译通过、算子可调、数值正确”到“单算子提速、整模质量达标”，每一步都是独立的工程门槛，实验记录需注明物理 GPU 型号、Compute Capability、工具链版本与输入几何。
