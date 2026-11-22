# 第三方归属与许可说明

原创文稿、图和示例的权利状态见 [COPYRIGHT](COPYRIGHT.md)。读者版实际包含以下第三方离线渲染资源。

| 资源 | 固定来源 | 保留的许可 |
|---|---|---|
| KaTeX 0.16.8 JS/CSS、auto-render | [官方 v0.16.8](https://github.com/KaTeX/KaTeX/tree/v0.16.8) | [MIT](assets/vendor/katex-0.16.8/LICENSE)，Khan Academy 及其他贡献者 |
| KaTeX 同版60个字体文件 | 官方 npm 原包 | [SIL OFL 1.1](assets/vendor/katex-0.16.8/OFL-1.1.txt)、[逐字体声明](assets/vendor/katex-0.16.8/FONT-NOTICES.txt)；含 Reserved Font Name |
| Mermaid 10.9.3 bundle | [官方 v10.9.3](https://github.com/mermaid-js/mermaid/tree/v10.9.3) | [MIT](assets/vendor/mermaid-10.9.3/LICENSE)、[依赖归属](assets/vendor/mermaid-10.9.3/BUNDLED-DEPENDENCY-NOTICES-r4.md)与[r6 内嵌代码许可补充](assets/vendor/mermaid-10.9.3/BUNDLED-EMBEDDED-NOTICES-r6.md) |

[固定资源清单](assets/vendor/manifest.json)登记原包、资源与许可的指纹；未分发 npm tarball、node_modules 或系统字体文件。PDF 的字体对象或子集由构建环境生成，字体与文稿的权利分别适用。

Mermaid bundle 中 DOMPurify 3.1.6 为 Apache-2.0 或 MPL-2.0 二选一，ELK 对应 elkjs 0.9.2/EPL-2.0。相邻归属文件保守收集106项固定依赖，也含可能未打入 bundle 的类型或 CLI 依赖，不作为实际执行 SBOM。`dagre-d3-es@7.0.10` 原包及对应 Git 树缺独立 LICENSE/NOTICE，所附记录保留原 dagre-d3 来源和 Chris Pettitt 的 MIT 文本，未补写 fork 新增贡献者授权。

r6 补充逐一核对 bundle 中保留的六段许可注释，新增 js-yaml 4.1.0 的[官方完整 MIT 许可](assets/vendor/mermaid-10.9.3/licenses/js-yaml-4.1.0/LICENSE)，并补充 Thenable、jQuery events、Bezier 与 Framer spring 片段的原版权、完整 MIT 条款及固定上游来源。原 bundle 字节与内嵌声明保持不变；直接源码未注明的 jQuery、Bezier 和 Framer 精确导入版本未被推定为已确认，所附参考许可的适用范围见[r6 说明](assets/vendor/mermaid-10.9.3/BUNDLED-EMBEDDED-NOTICES-r6.md)。

## 引用与外部依赖

论文和官方源码在结论附近链接，未复制论文全文或论文图片。PyTorch、FlashAttention、模型代码、模型权重与 wheel 是外部依赖，不随文稿包分发。正文中的 Mermaid 图由本书绘制。

SolarWM 代码、H3 等模型权重与输入素材的许可分别适用；模型名称、公开 URL、配置和哈希不授予权重或素材再分发权。实验摘录只提供指定结果与来源指纹。
