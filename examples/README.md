# 可执行案例

[附录 B](../docs/manuscript/appendix-b-reproduction.md)给出环境准备与运行顺序。这些案例使用合成输入，无需下载模型权重。

| 入口 | 演示内容 | 运行环境 |
|---|---|---|
| [review_public_evidence.py](review_public_evidence.py) | 重算保存的统计与几何 | Python 标准库 |
| [rtx_dit_lab.py](rtx_dit_lab.py) | Attention backend、CUDA Graph 和有界 raw KV | PyTorch；Attention/Graph 在 CUDA 运行，ring/list 在 CPU 运行 |
| [rtx_kv_transfer_lab.py](rtx_kv_transfer_lab.py) | pageable/pinned、提交方式和热路径 pinning | PyTorch、CUDA；固定的合成 Tensor Shape |

## 重算已有观测

```bash
python3 -B examples/review_public_evidence.py
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/measurements_sm90.json
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/ar_microbench_sm90.json
python3 -B -m unittest discover -s examples -p 'test_review_public_evidence.py' -v
python3 -B -m unittest discover -s examples -p 'test_attention_acceptance.py' -v
```

RTX 与 SM90 完整观测检查逐次样本、分位数、A/B/A 对照、数据大小及 Graph 回本调用数；AR 规模 KV 微基准仅保存 50 层 Gather 汇总耗时与池容量几何（`summary_only: true`），不含逐次热样本或独立生成脚本。GPU 性能复现与视频质量评估分别进行。

## CUDA 运行

设置 `DIT_LAB_ROOT` 为自己的可写绝对路径，并先创建输出目录。脚本检查父目录可写且至少剩余 1 GiB 空间。输出名称必须未被使用，脚本会拒绝覆盖。

```bash
mkdir -p "$DIT_LAB_ROOT/results"
python3 -B examples/rtx_dit_lab.py \
  --case all --device cuda:0 \
  --warmup 5 --samples 30 --output "$DIT_LAB_ROOT/results/operators.json"
python3 -B examples/rtx_kv_transfer_lab.py \
  --fixture tiny --device cuda:0 \
  --warmup 5 --samples 30 --output-dir "$DIT_LAB_ROOT/results/transfer-tiny" --pin-each-read
```

KV 搬运例先用 `tiny` 检查执行路径，再用 `--fixture large` 测较大负载。`large` 是教学用 Shape，与某个模型的 checkpoint 无关。保存的历史搬运例使用另一组显式 Shape（加上 `--pin-each-read` 以同时运行第 13 章表中的第三组对照），重放命令为：

```bash
python3 -B examples/rtx_kv_transfer_lab.py \
  --fixture explicit-shape --shape 1 5040 56 128 --device cuda:0 \
  --warmup 5 --samples 30 --output-dir "$DIT_LAB_ROOT/results/transfer-recorded-shape" --pin-each-read
```

输出默认保留 GPU 型号、Compute Capability、驱动版本等性能条件，省略 UUID、PCI bus ID 和 `CUDA_VISIBLE_DEVICES`。需要在本机核对物理设备时，可加 `--record-hardware-id`；在非 GeForce 数据中心显卡上运行可加 `--allow-non-rtx`；分享结果前应移除物理硬件标识。

Attention 固定 BF16 小例要求各输出对 FP32 参考、Flash 对 MATH 的最大绝对误差均不超过 0.005，relative L2 均不超过 0.005（0.5%），并要求 A1/A2 逐位相同。结果写入 `numerical_gate`；任一检查失败即为 `failed`。Flash backend 不可用且 MATH 基线正常时记录 `skipped`，不生成配对倍率，`technical_passed` 为 false，命令返回非零。该数值门仅适用于此固定 seed/shape 的教学输入，不代表完整模型或生成质量验收；其回归测试仅需 Python 标准库。

随包 `measurements.json` 与 `measurements_sm90.json` 中的 `source_sha256` 是采集当时的脚本指纹，与当前脚本文件的 SHA256 不同；RTX 5090 与 SM90 两次采集各自与随包脚本的差异见[附录 B](../docs/manuscript/appendix-b-reproduction.md#历史观测与随包脚本版本)。

计时条件与分析分别见 [Attention](../docs/manuscript/08-attention-kernels.md)、[Graph](../docs/manuscript/11-fusion-and-graphs.md)、[状态管理](../docs/manuscript/12-ar-state.md)和[搬运](../docs/manuscript/13-residency-and-transfer.md)章。将方法接入完整模型时，再固定权重、输入、求解器、输出长度和计时范围，测量端到端效果。
