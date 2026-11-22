# 附录 A 模型与实验配置

方法章使用两类材料：公开模型源码用于定位计算图与优化接入点，无需模型权重的合成实验用于隔离测量 Kernel、Launch、状态容器与数据搬运。这里集中列出模型版本与实验条件。

代表性模型按生成范式、源码可读性和状态协议选取，选型原则见第 1 章。

## Wan2.2：双向视频扩散

固定源码 revision 为 `1ea34ff48f87168174e12956e200b1d908b1c5ff`，核心入口为 T2V 的 `wan/text2video.py` 和 Block 实现 `wan/modules/model.py`。[官方源码](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff)

| 环节 | 固定源码中的行为 |
|---|---|
| 生成 | scheduler 循环更新完整视频 latent，视频 self-attention 与文本 cross-attention 分开 |
| expert | high-noise 与 low-noise 主干分别加载，按时间步和 boundary 选择 |
| CFG | 每步分别计算 conditional、unconditional 前向，再组合预测 |
| 条件 | 正、负文本先编码；各层和 expert 使用各自的投影权重 |
| 解码 | 去噪结束后在输出 rank 调用 VAE |
| 驻留 | 由 offload、dtype 和并行配置共同决定 |

各章的 Wan2.2 小节为源码接入分析；建立完整模型基线的参考命令见附录 B。

## SolarWM：连续 AR 的公开实现

[官方仓库](https://github.com/Junchao-cs/SolarWM)提供按 chunk 生成、历史 KV、滑动窗口及 clean refresh 的完整实现，对应第 12 章的状态生命周期分析。该仓库支持多种 backbone，具体的 latent 布局和时间坐标随 backend 而定。

[H3 模型页](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B)包含 base、Stage0.5、Stage1、Stage2 以及 Stage2 step 3900 EMA fix；[权重目录](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B/tree/main/SolarWM-h3-33B-sgf-stage2-158f-fix)列出了对应配置文件与 manifest（访问权重需接受模型许可条款）。

设每块生成 $C$ 个 latent 时间片，codec 的时间步幅为 $r$。由 $rC$ 换算输出像素帧数时，需核对首帧偏置、时间 padding、块间 overlap 与尾部裁剪。

## H3 与 ABot-World：架构对照

[H3 固定 README](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/README.md)用于对照混合视频生成架构；[ABot-World 论文 v1](https://arxiv.org/abs/2607.19191v1)用于对照双向 teacher、因果 student 与长程自回归训练机制。

## 合成实验与实测配置

随包实验数据采集自两套硬件环境，其中 GPU 算子、KV 搬运与容量微基准均使用单张 GPU（`cuda:0`），ring/list 容器对照在 CPU 上执行：

1. **RTX 5090**：NVIDIA GeForce RTX 5090（32 GB，SM120 Blackwell）、Linux x86_64、Python 3.10.12、PyTorch 2.8.0+cu128、CUDA runtime 12.8、driver 580.82.07，依赖锁文件固定 Triton 3.4.0。完整观测保存在 [measurements.json](evidence/measurements.json)。
2. **数据中心 SM90 GPU**：80 GB 级容量、Compute Capability 9.0（Hopper，架构属性参见 [NVIDIA Hopper 兼容性指南](https://docs.nvidia.com/cuda/hopper-compatibility-guide/)），Python 3.13.12、PyTorch 2.10.0、CUDA runtime 13.1、cuDNN 91900、driver 580.82.07。单卡算子与搬运完整样本保存在 [measurements_sm90.json](evidence/measurements_sm90.json)；50 层 AR 规模 KV 容量与 Gather 延迟汇总保存在 [ar_microbench_sm90.json](evidence/ar_microbench_sm90.json)。

| 案例 | 输入几何与对照路径 | 采样与计时方式 |
|---|---|---|
| Attention | BF16 Q/K/V `[1,8,512,64]`；MATH / FLASH / MATH；seed 42、dropout 0、无 mask | 每臂 5 warmup、30 hot，每次观测调用 10 次；分别记录 wall 与 Stream Event |
| CUDA Graph | BF16 `[64,512]`；8 组 Linear(512,512)+SiLU；eager / replay / eager；seed 43 | 捕获和 side-stream 预热单列，固定输入、权重和输出地址，含 NaN 覆写校验 |
| ring/list | CPU FP32，每份 K 或 V `[1,2,8,16]`、1 KiB；容量 5，12 次提交 | 每个 epoch 包含 clone/detach、按序读取、展平及 `torch.cat` 拼接 |
| Host→GPU 搬运 | BF16，每份 `[1,5040,56,128]`，10 份 tensor，共 722,534,400 B | 每臂 5 warmup、30 hot；单 Stream，完成 copy 后同步，隔离源内存与提交方式 |
| AR 规模 KV Gather | 50 层、56 头、128 维；5 块活跃窗口（6,000 token）vs 7 块分页池（8,512 槽位） | 仅记录 Chunk List 与 Python 侧 Paged BF16/FP8 物化的 50 层 Gather 汇总耗时及池容量构成（`summary_only: true`） |

前四项实验保存了全部 5 次 warmup 与 30 次 hot 逐次样本，以及采集时执行脚本的 `source_sha256`（与随包脚本的差异见[附录 B](appendix-b-reproduction.md#历史观测与随包脚本版本)）；第五项仅保存历史汇总结果与池容量几何，不含逐次热样本或独立生成脚本。重算与核验命令见附录 B。

## 常用符号对照表

全书在不同章节讨论张量几何、硬件存储与并行切分时，部分字母按领域惯例承担不同含义，阅读时可按下表对照上下文：

<!-- reader:table-may-split -->

| 符号 | 主要含义与出现章节 | 其他局部含义 |
|---|---|---|
| $B, T, H, W, C$ | 视频潜空间张量 `[B, C, T, H, W]` 的批大小、时序长度、空间高宽与潜变量通道数（第 2 章） | 第 2、8、14 章中 $H$ 亦指注意力头数；第 2 章与附录 A 中 $C$ 亦指单块 latent 时间片数；第 3 章 Roofline 中 $B$ 指访存字节数；第 5 章光流误差中 $C$ 指颜色通道数；第 11 章中 $C$ 指编译冷启动耗时 |
| $N, M, L_{\mathrm{KV}}$ | $N$ 为完整序列或单块 token 数，$M$ 为 AR 当前块 query token 数或 GEMM 行数，$L_{\mathrm{KV}}$ 为可见 KV 总长度（第 2、3、8、10 章） | 第 2 章 IO 复杂度中 $M$ 指片上快存容量；第 7 章 Taylor 余项中 $M$ 指二阶导上界；第 11 章 LUT 中 $M$ 指每层条件调制分支数；第 4、12 章中 $M_{\cdot}$ 指显存字节数 |
| $D, H, H_{\mathrm{kv}}, d$ | 隐藏层维度 $D$、注意力查询头数 $H$、KV 头数 $H_{\mathrm{kv}}$、单头维度 $d=D/H$（第 2、4、8、12 章） | 第 4 章显存账本中下标 $d$ 指设备编号；第 7 章中 $d_t$ 指特征相对距离；第 11 章冷启动公式中 $d$ 指单次请求节省秒数 |
| $K$ | AR 滑动窗口保留的历史 chunk 数（第 2、4、12 章，代码亦记 `W_history`） | Attention 中的 Key 矩阵（全书），或 GEMM $[M,K]\times[K,N]$ 的归约维（第 3、10 章） |
| $S$ | 去噪或采样的迭代步数（第 2、6、11 章；第 6 章另用 $S_h,S_l$ 分别计高、低噪声区间的时间点数） | 第 7 章 $S_t$ 为 TeaCache 累计判定量；第 8 章中 $S$ 为 FlashDecoding 的 KV 切分块数，$\Delta S$ 为 Q/K 平滑后的 score 补偿项；第 10 章中 $S$ 为量化缩放因子（$S_{\mathrm{block}}$ 为分块 scale），SVDQuant 中为对角平滑矩阵；第 15 章 $S_{\max}$ 为 Amdahl 加速上限，$S_{\mathrm{audio/frame}}$ 为每帧音频采样数；小写 $s$ 在第 6 章为 shift 系数，在第 12 章为环形池写入槽位 |
| $P$ | 序列并行（Ulysses / Ring）的 GPU 卡数（第 14 章） | 第 2 章中指每个 latent 时间片 patchify 后的视频 token 数；第 3 章 Roofline 中指硬件峰值计算吞吐；第 8 章中 $\widetilde P$ 指分块未归一化注意力权重 |
| $p_{\mathrm{drop}}, r, r_{\mathrm{area}}$ | 第 9 章中 $p_{\mathrm{drop}}$ 为稀疏注意力的被删位置概率质量，$r\in(0,1]$ 为 Token 压缩后的长度保留比例；第 15 章中 $r_{\mathrm{area}}\ge 1$ 为 VAE 瓦片解码覆盖面积比（重复计算比例为 $r_{\mathrm{area}}-1$） | 第 10 章 SVDQuant 中 $r$ 指低秩分支的秩（rank）；附录 A 中 $r$ 指 VAE 时间步幅 |
| $I$ | 第 3 章 Roofline 中的算术强度 $I=F/B$（FLOP/byte） | 第 5 章光流重投影中的视频帧 $I_t$；第 16 章 $2\times2$ 消融矩阵中的组合交互项 $I=T_{11}-T_{10}-T_{01}+T_{00}$ |
