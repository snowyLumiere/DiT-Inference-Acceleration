# 配套文件与复现说明

《DiT 推理加速：原理与工程实践》分为两篇、16 章和三份附录。总目录见 [README](README.md)，各篇目录见 [第一篇](docs/manuscript/part-1.md)和[第二篇](docs/manuscript/part-2.md)。Omni 服务框架另见配套文档 [vLLM-Omni 与 SGLang-Omni](docs/omni/inference-frameworks.md)，提供独立 [PDF](Omni_Inference_Frameworks.pdf) 和[离线 HTML](Omni_Inference_Frameworks.html)。

PDF、`build_whitepaper.html` 与配套目录应放在同一根目录。HTML 使用随包的 KaTeX、Mermaid 和字体，可离线阅读公式与图。PDF 中章节链接在文内跳转，其余本地链接打开配套文件；阅读器限制本地链接时，可按相同路径手动打开。

## 运行案例

先用 Python 标准库重算保存的观测，再选择 CUDA 小例：

```bash
python3 -B examples/review_public_evidence.py
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/measurements_sm90.json
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/ar_microbench_sm90.json
python3 -B -m unittest discover -s examples -p 'test_review_public_evidence.py' -v
```

CUDA 小例、固定依赖与模型入口见 [附录 B](docs/manuscript/appendix-b-reproduction.md)。保存数据覆盖 RTX 5090 与 SM90 单卡算子、搬运和 CPU 状态小例，以及 50 层 AR 规模 KV 容量与 Gather 微基准汇总。完整模型性能、多卡扩展和视频质量对照需另建基线。复现时为每次运行指定新输出路径。

## 核验下载完整性

在解压目录执行：

```bash
python3 -B scripts/verify_reader_package.py
```

校验器检查清单中的文件大小、SHA256 和文件集合，拒绝缺失、改动及多出的文件。也可执行 `shasum -a 256 -c SHA256SUMS`。清单用于核验文件完整性，不提供数字签名。

## 重建 PDF

安装 `requirements/pdf-build.txt` 中的依赖，提供 Chrome/Chromium 和所需中文字体，然后执行：

```bash
python3 -B scripts/build_whitepaper_pdf.py --check
python3 -B scripts/build_whitepaper_pdf.py
python3 -B scripts/build_whitepaper_pdf.py --omni --check
python3 -B scripts/build_whitepaper_pdf.py --omni
```

两个命令使用同一套本地渲染资源，输出主册及 `Omni_Inference_Frameworks.pdf`、对应 HTML 和各自的构建记录。失败时保留之前的 PDF。重建后需要重新校样，系统字体差异可能改变分页。
