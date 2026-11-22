# DiT 推理加速：原理与工程实践

> **副定位**：源码解析与合成实验 · **作者**：SnowyLumiere（[https://github.com/SnowyLumiere](https://github.com/SnowyLumiere)）· **版本**：v1.0（读者包 r9 · 2026-11-20）

本书从执行图、Tensor Shape 和测量边界出发，解释视频 DiT 与连续 AR-chunk 推理的性能瓶颈、加速方法和适用条件。面向有深度学习基础、希望实际分析和优化推理的算法工程师。

全书分两篇。第一篇建立生成范式、GPU、状态与测量基础；第二篇按方法把原理、实现和模型实践放在一起。Wan2.2 与 SolarWM 是双向扩散和连续 AR 的两条源码主线，H3 与 ABot-World 用于架构或训练对照。Omni 与框架方案另见[独立配套文档](docs/omni/inference-frameworks.md)。

[主册 PDF](DiT_Inference_Acceleration_Whitepaper_v1.0.pdf) · [离线 HTML](build_whitepaper.html) · [Omni 配套 PDF](Omni_Inference_Frameworks.pdf) · [配套使用说明](RELEASE_DIT_INFRA.md) · [代码入口](examples/README.md)

## 总目录

每篇另有入口目录：[第一篇](docs/manuscript/part-1.md)、[第二篇](docs/manuscript/part-2.md)。PDF 提供可点击总目录、篇内目录和分层书签。

## 第一篇：计算、硬件与测量基础

- [第1章 生成范式与模型地图](docs/manuscript/01-paradigms.md)
- [第2章 从输入到输出的执行图与工作量](docs/manuscript/02-execution-graph.md)
- [第3章 GPU 执行与性能约束](docs/manuscript/03-gpu-execution.md)
- [第4章 状态、显存与数据生命周期](docs/manuscript/04-memory-and-state.md)
- [第5章 性能分析与可复现实验](docs/manuscript/05-measurement.md)

## 第二篇：加速方法与模型实践

- [第6章 减少网络求值：求解器、少步、CFG 与蒸馏](docs/manuscript/06-evaluations-and-distillation.md)
- [第7章 跨步特征复用与自适应缓存](docs/manuscript/07-feature-cache.md)
- [第8章 Dense Attention Kernel 与低精度 Attention](docs/manuscript/08-attention-kernels.md)
- [第9章 稀疏、Token 压缩与替代 Attention](docs/manuscript/09-sparsity-and-tokens.md)
- [第10章 Linear/GEMM 与低精度计算](docs/manuscript/10-gemm-and-quantization.md)
- [第11章 算子融合、预计算、编译与 CUDA Graph](docs/manuscript/11-fusion-and-graphs.md)
- [第12章 AR 历史 KV 与状态管理](docs/manuscript/12-ar-state.md)
- [第13章 驻留、Offload 与数据搬运](docs/manuscript/13-residency-and-transfer.md)
- [第14章 多 GPU 并行与通信](docs/manuscript/14-parallelism.md)
- [第15章 VAE、流式执行与端到端交付](docs/manuscript/15-decoding-and-streaming.md)
- [第16章 组合优化与回归验收](docs/manuscript/16-composition-and-validation.md)

## 延伸与附录

- [延伸阅读指引](docs/manuscript/extension-omni.md)
- [附录 A 模型与实验配置](docs/manuscript/appendix-a-model-contracts.md)
- [附录 B 复现入口与操作步骤](docs/manuscript/appendix-b-reproduction.md)
- [附录 C 来源与实验数据](docs/manuscript/appendix-c-evidence.md)

## 如何使用案例

公开模型小节沿源码解释接入位置。实验覆盖 RTX 5090 与数据中心 SM90 GPU（80 GB）上的单卡 Attention、CUDA Graph、KV 搬运，以及 CPU 状态小例；另提供 50 层 AR 规模 KV 容量与 Gather 延迟的合成微基准汇总。各实验的设备配置、样本结构及适用边界见附录 A、C。整模加速与生成质量需另建匹配基线，目前未收录整模性能或多卡扩展对照。

[来源与实验索引](docs/manuscript/appendix-c-evidence.md)列出原始依据；[复现入口](docs/manuscript/appendix-b-reproduction.md)从标准库重算到 CUDA 小例逐步展开。第二篇的篇内目录另附 AR-chunk 阅读路线，方便沿求值、历史状态、搬运和交付串联各章。

## 配套范围、引用与勘误

读者版按明确文件清单收录正文、代码、字段筛选后的实验摘录和离线渲染资源。模型权重、输入素材、视频、完整 trace、旧案例与运维记录不随包分发。当前仅提供阅读与学习参考，未新增再分发、修改或再许可授权，详见 [COPYRIGHT](COPYRIGHT.md)；第三方资源归属见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

- **公开资料核对截止**：2026-11（各固定源码以正文及附录 C 所列 commit 为准）。
- **推荐引用**：`SnowyLumiere. 《DiT 推理加速：原理与工程实践（源码解析与合成实验）》, v1.0 (Reader Edition r9), 2026.`
- **勘误与反馈**：可通过作者 GitHub 主页 [https://github.com/SnowyLumiere](https://github.com/SnowyLumiere) 提交反馈或追踪后续版本。
