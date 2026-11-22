# 从这里开始

- **作者**：SnowyLumiere（[https://github.com/SnowyLumiere](https://github.com/SnowyLumiere)）
- **版本对应**：内容版本 `v1.0` · 当前读者包 `r9`（2026-11-20）。本次技术校订、封面白块修复、署名水印与验证范围见 [r9 发布说明](RELEASE_NOTES_R9.md)（历史修订另见 [r7](RELEASE_NOTES_R7.md)、[r6](RELEASE_NOTES_R6.md)）。
- **定位与使用权限**：面向有深度学习基础工程师的《DiT 推理加速：原理与工程实践（源码解析与合成实验）》。原创文稿、图与示例代码保留版权，当前仅供阅读与学习参考，未新增再分发或修改授权，详见 [COPYRIGHT.md](COPYRIGHT.md)。

PDF 中流程图已整体嵌入，避免 Zotero“阅读模式”将图拆散；正文各页已加入不干扰复制与重排的轻量归属水印。阅读模式仍可能遗漏行内公式，核对公式请使用普通 PDF 页面视图或离线 HTML。

主册：[DiT 推理加速：原理与工程实践](DiT_Inference_Acceleration_Whitepaper_v1.0.pdf)（[离线 HTML](build_whitepaper.html)）。

独立配套：[Omni 推理框架](Omni_Inference_Frameworks.pdf)（[离线 HTML](Omni_Inference_Frameworks.html)），讨论 Omni 与 DiT、vLLM-Omni 和 SGLang-Omni。

[README](README.md) 提供总目录、引用与勘误入口、Markdown 入口和配套代码说明。

正文结合原理推导、公开论文与源码分析，以及 RTX 5090 和数据中心 SM90 GPU（80 GB）的合成实验。算子与搬运例使用单张 GPU，并提供 50 层 AR 规模 KV 的容量与 Gather 历史汇总观察（`summary_only`）。公开模型接入分析与整模测量分开呈现，包内未收录整模性能、多卡扩展或视频质量实测。

校验解压内容：`python3 -B scripts/verify_reader_package.py`。

重算实验数据：`python3 -B examples/review_public_evidence.py`；SM90 数据的核验命令见附录 B。
