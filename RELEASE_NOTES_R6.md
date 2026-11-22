# 读者版 r6 发布说明

发布日期：2026-11-15。基于 `DiT_Reader_Edition_20261104_r5.zip` 修复并重新打包。

## 本次修复

- 补入 js-yaml 4.1.0 的官方完整 MIT 许可与版权声明，并补齐 Mermaid bundle 内 Thenable、jQuery events、Bezier、Framer spring 片段的许可材料和来源。详情见[第三方说明](THIRD_PARTY_NOTICES.md)与[r6 内嵌代码许可补充](assets/vendor/mermaid-10.9.3/BUNDLED-EMBEDDED-NOTICES-r6.md)。Mermaid、KaTeX 的执行资源字节未变。
- 修复 [Attention 教学脚本](examples/rtx_dit_lab.py)的数值验收：各必需比较要求输出有限、最大绝对误差与 Relative L2 均不超过 0.005，A1/A2 必须逐位相同。该门限只针对固定 BF16 教学输入；明显错误的有限值输出不再通过。Flash backend 不可用时标记 `skipped`，不记为 `technical_passed`。
- 新增[标准库回归测试](examples/test_attention_acceptance.py)，覆盖历史观测、100% 相对误差、非有限值、超阈、缺失字段与跳过状态。运行方法见[案例说明](examples/README.md)。
- 更新[附录 B](docs/manuscript/appendix-b-reproduction.md)与两份历史观测的导出说明，明确 r5 与 r6 的脚本差异；第 12 章的分页 KV 选型结论改为受实验范围约束的建议。
- 重新构建两份 PDF 与离线 HTML，并据实际构建结果更新 `READER_BUILD.json`、`OMNI_BUILD.json`、vendor 清单、`READER_MANIFEST.json` 与 `SHA256SUMS`。构建依赖版本见[依赖清单](requirements/pdf-build.txt)。

## 验证范围

29 项 examples 测试与 59 项 scripts 测试通过。三份历史实验数据的重算通过；历史计时、比较结果、输入指纹与采集脚本指纹保持不变，仅更新导出说明。新数值门可接受两套已保存的 Attention 比较数据。

两份离线 HTML 的公式、流程图与内部锚点检查通过；重新生成的 PDF 经过全页排版总览及几何检查。包内全部文件可通过以下标准库命令核验：

```bash
python3 -B scripts/verify_reader_package.py
```

本次没有重新执行 CUDA 实验、整模推理或生成质量评测。文中性能数值仍是已保存的历史合成实验，不是 r6 新采集结果；完整模型性能、多卡扩展与视频质量需另建基线。

版权与使用范围沿用 [COPYRIGHT.md](COPYRIGHT.md)。本包未新增统一开源许可。
