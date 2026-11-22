# 附录 C 来源与实验数据

本书将理论公式推导、公开模型源码分析与可复现的合成微基准结合在一起：代数公式界定计算量与访存下界，固定 commit 的开源代码定位工程接入点，合成实验则隔离测量单一算子、状态容器或跨设备搬运机制。附录 A 汇总了实验软硬件条件，附录 B 给出了从标准库重算到完整模型基线的操作步骤。

## 来源索引

[来源清单](evidence/primary-sources.json)登记了本书重点核对的开源源码与仓库文档快照（含固定 Git revision 与 SHA256 指纹）；其余随文引用的固定 commit 源码、公开论文与 NVIDIA、PyTorch 等官方技术手册均直接在正文对应结论处通过外链给出。

| 内容 | 主要阅读入口 |
|---|---|
| 双向执行图 | [Wan2.2 T2V](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/text2video.py)、[Block](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/model.py) |
| GPU 与 GEMM | [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)、[Matrix Multiplication](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) |
| 求解与蒸馏 | [Flow Matching](https://arxiv.org/abs/2210.02747)、[DPM-Solver++](https://arxiv.org/abs/2211.01095)、[UniPC](https://arxiv.org/abs/2302.04867)；CausVid / Self Forcing 见第 6 章 |
| Attention 与缓存 | [FlashAttention](https://arxiv.org/abs/2205.14135)、[FA3](https://tridao.me/blog/2024/flash3/)、[TeaCache](https://arxiv.org/abs/2411.19108)；稀疏实现见第 9 章 |
| 低精度 | [SVDQuant](https://arxiv.org/abs/2411.05007)、[NVFP4 格式](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)；工具链及版本差异见第 10 章 |
| 编译与执行 | [PyTorch 2.8 CUDA 语义](https://docs.pytorch.org/docs/2.8/notes/cuda.html)、[Regional compilation](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html) |
| AR 状态 | [SolarWM](https://github.com/Junchao-cs/SolarWM)、[ABot-World 论文 v1](https://arxiv.org/abs/2607.19191v1) |
| Omni 框架 | 独立配套文档 [Omni 推理框架](../omni/inference-frameworks.md)中的固定源码链接 |

## 实验数据如何使用

1. [measurements.json](evidence/measurements.json) 与 [measurements_sm90.json](evidence/measurements_sm90.json) 分别保存了 RTX 5090 与数据中心 SM90 GPU 环境下的 Attention backend、CUDA Graph、CPU ring/list 容器以及 Host→GPU KV 搬运的全部 warmup/hot 逐次观测、统计分位数、输入张量指纹、显存高水位与数值误差对照。
2. [ar_microbench_sm90.json](evidence/ar_microbench_sm90.json) 保存了 50 层 AR 视频规模（56 头、128 维、每块 1200 token）下活跃窗口（5 块 6,000 token，Gather 输出 8.60 GB BF16）与静态分页池（7 块 8,512 槽位，含页尾 padding 与 FP8 scales）的容量分解，以及连续块 `torch.cat` 与 Python 侧 Paged BF16 / Paged FP8 逐页物化路径的 50 层 Gather 汇总耗时（`summary_only: true`，无逐次热样本或随包生成脚本）。

三份数据文件均可通过 `python3 -B examples/review_public_evidence.py`（配合 `--input` 参数）离线核验。有效搬运带宽定义为单轮净载荷字节数除以包含提交与同步在内的 Host wall time，用于衡量应用层实际可得的传输吞吐。

阅读实验数据时，按各章对应的工程问题对号入座：第 8 章对比 Attention 算子后端，第 11 章衡量小算子栈的发射开销与 Graph 摊销回本点，第 12 章对比滑动窗口状态容器与分页物化成本，第 13 章评估锁页内存对 PCIe 搬运的影响。

## 后续实验扩展

在目标业务模型上开展整模评测时，可从 Wan2.2 与公开连续 AR 模型的原生基线出发：先冻结测试集与原生参考输出，按第 5 章设定质量门禁，再按第 16 章的 $2\times2$ 矩阵依次报告单请求延迟、首包/稳态块间隔、显存峰值与长序列视频质量。
