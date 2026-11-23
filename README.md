# DiT 推理加速：原理与工程实践

> **English TL;DR** *(prose in Chinese; all code, JSON receipts, stdlib verifiers & pinned refs are universal)*:  
> A free 63-page engineering handbook on **Video DiT & Continuous AR-Chunk Inference Acceleration** ([Wan2.2](https://github.com/Wan-Video/Wan2.2) & [SolarWM](https://github.com/Junchao-cs/SolarWM) pinned-commit source walkthroughs, Hopper SM90 vs Blackwell SM100/SM120 RTX 5090 migration checklist, reproducible CUDA microbenchmarks with honest negative results) + a 7-page standalone deep-dive on **vLLM-Omni & SGLang-Omni**.  
> **Download**: [Release v1.0-r9 ZIP + SHA256](https://github.com/snowyLumiere/DiT-Inference-Acceleration/releases/tag/v1.0-r9) · [Main PDF (63p)](DiT_Inference_Acceleration_Whitepaper_v1.0.pdf) · [Omni Companion PDF (7p)](Omni_Inference_Frameworks.pdf) · [Offline Verify (`python3 -B examples/review_public_evidence.py`)](examples/README.md)

> **副定位**：源码解析与合成实验 · **作者**：[SnowyLumiere](https://github.com/snowyLumiere) · **版本**：`v1.0`（读者包 `r9` · 2026-11-20）

[![Release v1.0-r9](https://img.shields.io/badge/Release-v1.0--r9-2563eb?style=flat-square)](https://github.com/snowyLumiere/DiT-Inference-Acceleration/releases/tag/v1.0-r9)
[![Main PDF](https://img.shields.io/badge/主册_PDF-63页-0f172a?style=flat-square)](DiT_Inference_Acceleration_Whitepaper_v1.0.pdf)
[![Omni Companion](https://img.shields.io/badge/Omni_配套-7页-0369a1?style=flat-square)](Omni_Inference_Frameworks.pdf)
[![Discussions](https://img.shields.io/badge/Discussions-勘误与投票-16a34a?style=flat-square)](https://github.com/snowyLumiere/DiT-Inference-Acceleration/discussions)

<table>
  <tr>
    <td width="240" valign="top" align="center">
      <a href="DiT_Inference_Acceleration_Whitepaper_v1.0.pdf">
        <img src=".github/assets/cover-preview.png" width="220" alt="《DiT 推理加速：原理与工程实践》封面" />
      </a>
    </td>
    <td valign="top">
      <p>本书面向有深度学习基础、希望实际分析和优化推理的算法与 AI Infra 工程师，从<strong>执行图、Tensor Shape、显存生命周期与测量边界</strong>出发，系统拆解视频 DiT 与连续 AR-chunk 推理的性能瓶颈、加速方法与适用条件。</p>
      <p><strong>立即获取（推荐下载完整离线包）</strong></p>
      <ul>
        <li>📦 <strong><a href="https://github.com/snowyLumiere/DiT-Inference-Acceleration/releases/tag/v1.0-r9">下载完整读者包（Release v1.0-r9 ZIP + SHA256）</a></strong> — 含两份 PDF、离线 HTML（内置 KaTeX/Mermaid）、全套源码与实验数据</li>
        <li>📘 <strong><a href="DiT_Inference_Acceleration_Whitepaper_v1.0.pdf">主册 PDF（63 页）</a></strong> · <a href="build_whitepaper.html">主册离线 HTML</a></li>
        <li>📗 <strong><a href="Omni_Inference_Frameworks.pdf">Omni 推理框架独立配套 PDF（7 页）</a></strong> · <a href="Omni_Inference_Frameworks.html">Omni 离线 HTML</a></li>
        <li>🧪 <strong><a href="examples/README.md">合成实验与标准库离线重算脚本</a></strong> · <a href="RELEASE_NOTES_R9.md">r9 发布说明</a></li>
      </ul>
      <p><strong>为什么读这本？</strong></p>
      <ul>
        <li><strong>两条固定 commit 源码主线</strong>：以 Wan2.2（双向双 expert）与 SolarWM / Self Forcing / CausVid（连续 AR-chunk）贯通全书接入点，而非停留在论文摘要。</li>
        <li><strong>SM90 → Blackwell (SM100 / SM120) 硬件视角</strong>：明确 Hopper、B200 与 RTX 5090 在 MMA/TMEM、Shared Memory 与 P2P 上的工程差异。</li>
        <li><strong>诚实披露实验边界与负结果</strong>：保留 CPU ring 略慢、热路径临时 <code>pin_memory()</code> 带宽掉半、Python 侧分页 FP8 Gather 耗时飙升等反直觉观测；不把单卡合成实验包装成整模成绩。</li>
        <li><strong>附带多阶段框架深挖</strong>：独立配套文档按固定源码拆解 vLLM-Omni 与 SGLang-Omni 的阶段调度、<code>async_chunk</code>、NIXL / CUDA IPC Relay、背压与取消。</li>
      </ul>
    </td>
  </tr>
</table>

---

## 快速导航

[从这里开始](READ_FIRST.md) · [主册 PDF](DiT_Inference_Acceleration_Whitepaper_v1.0.pdf) · [离线 HTML](build_whitepaper.html) · [Omni 配套 PDF](Omni_Inference_Frameworks.pdf) · [配套使用说明](RELEASE_DIT_INFRA.md) · [代码入口](examples/README.md) · [勘误与讨论](https://github.com/snowyLumiere/DiT-Inference-Acceleration/discussions)

全书分两篇。第一篇建立生成范式、GPU、状态与测量基础；第二篇按方法把原理、实现和模型实践放在一起。Wan2.2 与 SolarWM 是双向扩散和连续 AR 的两条源码主线，H3 与 ABot-World 用于架构或训练对照。Omni 与框架方案另见[独立配套文档](docs/omni/inference-frameworks.md)。

## 总目录

每篇另有入口目录：[第一篇](docs/manuscript/part-1.md)、[第二篇](docs/manuscript/part-2.md)。PDF 提供可点击总目录、篇内目录和分层书签（已针对 Zotero 10 流程图完整保留与 macOS/iOS 渲染实测验证）。

### 第一篇：计算、硬件与测量基础

- [第1章 生成范式与模型地图](docs/manuscript/01-paradigms.md)
- [第2章 从输入到输出的执行图与工作量](docs/manuscript/02-execution-graph.md)
- [第3章 GPU 执行与性能约束](docs/manuscript/03-gpu-execution.md)
- [第4章 状态、显存与数据生命周期](docs/manuscript/04-memory-and-state.md)
- [第5章 性能分析与可复现实验](docs/manuscript/05-measurement.md)

### 第二篇：加速方法与模型实践

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

### 延伸与附录

- [延伸阅读指引](docs/manuscript/extension-omni.md)
- [附录 A 模型与实验配置](docs/manuscript/appendix-a-model-contracts.md)
- [附录 B 复现入口与操作步骤](docs/manuscript/appendix-b-reproduction.md)
- [附录 C 来源与实验数据](docs/manuscript/appendix-c-evidence.md)

---

## 如何使用案例与离线核验

公开模型小节沿固定 commit 源码解释接入位置。实验覆盖 RTX 5090（SM120）与数据中心 SM90 GPU（80 GB）上的单卡 Attention、CUDA Graph、KV 搬运，以及 CPU 状态小例；另提供 50 层 AR 规模 KV 容量与 Gather 延迟的合成微基准历史汇总（`summary_only`）。各实验的设备配置、样本结构及适用边界见附录 A、C。整模加速与生成质量需另建匹配基线，目前未收录整模性能或多卡扩展对照。

无需 GPU，仅用 Python 标准库即可重算保存的实验统计与几何：

```bash
python3 -B examples/review_public_evidence.py
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/measurements_sm90.json
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/ar_microbench_sm90.json
python3 -B -m unittest discover -s examples -p 'test_*.py' -v
```

> 注：解压 [Release `v1.0-r9` 压缩包](https://github.com/snowyLumiere/DiT-Inference-Acceleration/releases/tag/v1.0-r9) 后，可执行 `python3 -B scripts/verify_reader_package.py` 校验全部 243 个发行文件的大小与 SHA256（Git 仓库根目录额外包含 GitHub 展示用的 `.github/` 预览图与首页排版，因此包完整性校验以 Release 压缩包为准）。

[来源与实验索引](docs/manuscript/appendix-c-evidence.md)列出原始依据；[复现入口](docs/manuscript/appendix-b-reproduction.md)从标准库重算到 CUDA 小例逐步展开。第二篇的篇内目录另附 AR-chunk 阅读路线，方便沿求值、历史状态、搬运和交付串联各章。

---

## 相关开源项目（SnowyLumiere）

如果你也在做 DiT、自回归视频模型或注意力掩码/算子工程，可以一并参考以下配套工具：

- **[driftguard-dit](https://github.com/snowyLumiere/driftguard-dit)**：面向自回归流式 DiT 与世界模型的零训练 latent 漂移监测、早停熔断与自适应去噪校准工具（对应第 5、6、12 章长程漂移问题）。
- **[MaskBench-Viz](https://github.com/snowyLumiere/MaskBench-Viz)**：将多模态注意力规则映射到物理计算 Tile 的交互式可视化器，用于检查块跳过、逐元素掩码与因果边界（对应第 2、8、9 章）。

---

## 配套范围、引用与勘误

读者版按明确文件清单收录正文、代码、字段筛选后的实验摘录和离线渲染资源。模型权重、输入素材、视频、完整 trace、旧案例与运维记录不随包分发。当前仅提供阅读与学习参考，未新增再分发、修改或再许可授权，详见 [COPYRIGHT](COPYRIGHT.md)；第三方资源归属见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

- **公开资料核对截止**：2026-11（各固定源码以正文及附录 C 所列 commit 为准）。
- **推荐引用**：`SnowyLumiere. 《DiT 推理加速：原理与工程实践（源码解析与合成实验）》, v1.0 (Reader Edition r9), 2026.`
- **勘误、讨论与 v1.1 建议**：欢迎前往 [GitHub Discussions](https://github.com/snowyLumiere/DiT-Inference-Acceleration/discussions) 提交勘误，或参与下一版真实模型端到端案例投票。
