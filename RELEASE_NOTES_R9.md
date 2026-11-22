# 读者版 r9（v1.0 公开发布修订、署名与水印）

- **作者**：SnowyLumiere（[https://github.com/SnowyLumiere](https://github.com/SnowyLumiere)）
- **发布日期**：2026-11-20
- **版本对应**：内容版本 `v1.0` · 读者包 `r9`（基于 `DiT_Reader_Edition_20261116_r8.zip` 修订并重新构建；`r8` 相对 `r7` 仅在 `scripts/test_build_whitepaper_pdf.py` 为一项 PyMuPDF 测试补充了 `skipUnless` 装饰器并更新清单）

## 本次修复与更新

### 1. 封面渲染缺陷与跨平台版式修复
- **修复 macOS / iOS 封面白块缺陷**：移除主册封面 `.cover-title` 的 `-webkit-background-clip: text` 渐变裁剪，改为纯色文字，彻底消除 Apple PDFKit / 预览（Preview）/ QuickLook / Safari 在第 1 页标题处渲染出实心白块及 PyMuPDF 残留边框的问题。
- **消除末尾孤页与页内排版优化**：微调页边距与行距后，主册由 65 页收整为 63 页、Omni 配套由 8 页收整为 7 页，两份文档的阅读说明、总目录与末尾章节均无孤页。
- **补齐离线 HTML 屏幕阅读样式**：为两份离线 HTML 补充 `<meta name="viewport">`、`@media screen` 居中阅读栏宽、窄屏表格/公式横向滚动与轻量屏幕底纹，不影响 `@media print` 打印分页。

### 2. 技术表述校订（6+1 处）
- **第 8 章（Sage2 / Sage3 的 $\mu_Q$ 平滑粒度）**：明确 SageAttention2 v2 算法描述在分块前对整张 $Q$ 求逐通道均值，SageAttention3 v3 按每个 query block 求 $\mu_{Q_i}$，并统一表格与正文中对 Q/K/V 平滑的描述。
- **第 10 章（SM120 Shared Memory 口径）**：将 Hopper/Blackwell 核对表拆分为“统一 L1/SMEM 池”（256 / 256 / 128 KiB）与“可配置 Shared Memory 上限”（SM90/SM100 每 SM 228 KiB、单 Block 227 KiB；SM120 每 SM 100 KiB、单 Block 99 KiB），并注明 NVIDIA 不同文档间的口径差异与实测查询方式。
- **第 12 章（AR 规模微基准证据分层）**：以独立提示框标明 50 层 AR Gather 记录为 `summary_only` 历史汇总观察，与保留全部逐次热样本和采集脚本的实验区分呈现。
- **第 14 章（CFG Parallel 与 AR KV Offload 前置条件）**：修正 CFG Parallel 为“单卡计算负载与理想关键路径接近减半（总 FLOPs 不变）”；明确连续 AR 负载应优先单卡驻留与局部算子优化，仅在显存受限时才评估持久锁页 KV 卸载与预取。
- **第 15 章（TTFF 边界）**：区分客户端“首帧呈现延迟（TTFF）”与“服务端首块就绪时间”，明确公式起止边界及 $T_{\text{DiT}}(\text{Chunk 0})$ 是否计入 clean KV commit。
- **第 16 章（clean refresh 适用范围）**：将“不可跳过 clean refresh”限定至采用独立历史提交前向协议的连续 AR 模型，与第 12 章保持一致。
- **Omni 配套与附录 C**：修正 `OmniScheduler._event_loop_overlap` 方法名；清理 [来源清单](docs/manuscript/evidence/primary-sources.json) 中 `xdit-project/xDiT` 重复的旧 commit 条目，并在附录 C 准确界定集中清单的登记范围。

### 3. 署名、四层归属水印与测试补齐
- **明确署名与版权归属**：在封面、第 2 页版本/引用/勘误栏、Omni 配套页眉、[README.md](README.md)、[READ_FIRST.md](READ_FIRST.md)、[COPYRIGHT.md](COPYRIGHT.md) 及两份 PDF 的 `Author` 元数据中统一署名 `SnowyLumiere`（仍剥离 `creator`、`producer`、构建时间戳、本机路径与 XMP 元数据）。
- **页脚 + 非文本斜向底纹水印**：每页页脚固定显示 `© 2026 SnowyLumiere (github.com/SnowyLumiere)`；正文各页底层（`overlay=False`）以共享透明 PNG Image XObject 铺设低透明度斜向底纹水印，离线 HTML 在屏幕阅读模式下同步显示轻量 SVG 底纹。该实现不引入可选中水印文字，不干扰复制粘贴与 Zotero 阅读模式段落提取。
- **修复标准库单测漏装饰器**：在 [scripts/test_build_whitepaper_pdf.py](scripts/test_build_whitepaper_pdf.py) 为 `test_only_marked_tables_may_split_across_pages` 补加 `Markdown required` 跳过条件，并新增封面样式、作者元数据与非文本水印回归测试。

## 验证范围

- **自动化测试**：完整构建环境下 `scripts` 68 项测试、`examples` 29 项测试全部通过；无第三方依赖的标准库 Python 环境下 `scripts` 与 `examples` 测试零报错通过（依赖项按预期跳过）。三份历史实验数据的离线重算与几何核验全部通过。
- **跨渲染器视觉与 Zotero 兼容验证**：使用 macOS Apple PDFKit 核验两份 PDF 封面、目录、表格、公式、流程图与底纹水印渲染正常；使用 Zotero 10.0.9 自带 PDF.js 5.7.0 / ONNX document-worker 复核阅读模式结构提取，主册 16/16 与 Omni 1/1 流程图完整保留（最小覆盖率 99.99997%，图内散落段落为 0，全页水印误判为 0），详见 [兼容性记录](PDF_COMPATIBILITY_R9.json)。
- **完整性与隐私门禁**：包内 243 个文件通过 `reader_privacy` 全量扫描（含 PDF 文本、元数据、书签、注释与链接）、本地相对链接检查及标准库清单校验：

```bash
python3 -B scripts/verify_reader_package.py
```

本次没有重新执行 GPU 实验、整模推理或视频质量评测；文中性能数值仍为已保存的历史合成实验，历史观测与随包脚本的差异说明沿用附录 B。版权与使用范围见 [COPYRIGHT.md](COPYRIGHT.md)，第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
