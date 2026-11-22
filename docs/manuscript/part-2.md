# 第二篇：加速方法与模型实践

各章围绕一种方法组织：先解释减少了什么工作，再对照双向模型与 AR 的输入、状态和接入条件。读完单项方法后，第 16 章将它们组合成一套可重复的实施与回归流程。

- [第6章 减少网络求值：求解器、少步、CFG 与蒸馏](06-evaluations-and-distillation.md)
- [第7章 跨步特征复用与自适应缓存](07-feature-cache.md)
- [第8章 Dense Attention Kernel 与低精度 Attention](08-attention-kernels.md)
- [第9章 稀疏、Token 压缩与替代 Attention](09-sparsity-and-tokens.md)
- [第10章 Linear/GEMM 与低精度计算](10-gemm-and-quantization.md)
- [第11章 算子融合、预计算、编译与 CUDA Graph](11-fusion-and-graphs.md)
- [第12章 AR 历史 KV 与状态管理](12-ar-state.md)
- [第13章 驻留、Offload 与数据搬运](13-residency-and-transfer.md)
- [第14章 多 GPU 并行与通信](14-parallelism.md)
- [第15章 VAE、流式执行与端到端交付](15-decoding-and-streaming.md)
- [第16章 组合优化与回归验收](16-composition-and-validation.md)

连续 AR 可以按问题串读：用[第6章](06-evaluations-and-distillation.md)确定每块的去噪与提交次数，用[第12章](12-ar-state.md)设计历史读写，再沿[第13章](13-residency-and-transfer.md)和[第15章](15-decoding-and-streaming.md)处理驻留、搬运与交付。随后回到 Attention、量化和 Graph 各章，检查这些方法是否保留了相同的状态语义。这样可把 AR 的完整路径连起来，也便于在同一方法下直接比较双向与 AR 的差异。

[返回总目录](../../README.md)
