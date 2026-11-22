# 附录 B 复现入口与操作步骤

复现实验按三步展开：先用 Python 标准库重算并核验随包保存的实验数据，再在本地 GPU 上运行无需模型权重的算子与搬运小例，最后在完整模型上建立配对基线。

## 无 GPU：重算实验数据

仅需 Python 标准库即可执行全部三份证据文件的重算与几何核验：

```bash
# 重算 RTX 5090 单卡算子与搬运数据
python3 -B examples/review_public_evidence.py

# 重算数据中心 SM90 单卡算子与搬运数据
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/measurements_sm90.json

# 核验 50 层 AR 规模 KV 容量几何与 Gather 汇总数据
python3 -B examples/review_public_evidence.py --input docs/manuscript/evidence/ar_microbench_sm90.json
```

前两项命令逐样本重算 30 次 hot 观测的均值、中位数、分位数、首尾漂移、A/B 加速比、有效搬运带宽及 CUDA Graph 回本调用数；第三项命令核验 50 层活跃窗口、Gather 输出 BF16 载荷与 7 块分页池（含页尾 padding 与 FP8 scales）的字节算术及 Gather 倍率（该微基准为 `summary_only: true` 历史汇总，不含逐次热样本或随包生成脚本）。

## CUDA 小例：先确认设备和输出目录

RTX 5090 数据运行于 Linux x86_64、Python 3.10、PyTorch 2.8.0+cu128；SM90 数据运行于 Python 3.13.12、PyTorch 2.10.0 与 CUDA runtime 13.1。在其他兼容环境中运行时，先记录实际软硬件版本，并通过 profiler 确认命中的算子 backend。

将环境变量 `DIT_LAB_ROOT` 指向本地可写的绝对工作目录（父目录需已存在且剩余空间不少于 1 GiB；脚本拒绝覆盖已存在的输出路径）：

```bash
: "${DIT_LAB_ROOT:?设置为自己的绝对工作目录}"
mkdir -p "$DIT_LAB_ROOT/results"
python3 -B examples/rtx_dit_lab.py \
  --case all --device cuda:0 \
  --warmup 5 --samples 30 --output "$DIT_LAB_ROOT/results/operators.json"
python3 -B examples/rtx_kv_transfer_lab.py \
  --fixture tiny --device cuda:0 \
  --warmup 5 --samples 30 --output-dir "$DIT_LAB_ROOT/results/transfer-tiny" --pin-each-read
```

`tiny` 通过后，可用 `--fixture large` 运行通用教学尺寸 `[1,4096,32,128]`（单张 BF16 张量 33,554,432 B，10 张合计 335,544,320 B）。若要重放第 13 章及附录 A 保存的显式几何尺寸 `[1,5040,56,128]`（单张 72,253,440 B，10 张合计 722,534,400 B），可执行：

```bash
python3 -B examples/rtx_kv_transfer_lab.py \
  --fixture explicit-shape --shape 1 5040 56 128 --device cuda:0 \
  --warmup 5 --samples 30 --output-dir "$DIT_LAB_ROOT/results/transfer-recorded-shape" --pin-each-read
```

脚本内置了 NaN 毒化覆写、张量版本号与逐位等价检查，并将初始化耗时与热态计时分开保存。默认输出省略 GPU UUID 与 PCI bus ID；如需在本机排查多卡物理拓扑可加 `--record-hardware-id`，在非 GeForce 数据中心显卡上运行可加 `--allow-non-rtx`。

### 历史观测与随包脚本版本

`measurements.json` 与 `measurements_sm90.json` 中的 `source_sha256` 记录的是采集当时的脚本指纹，与当前随包 `rtx_dit_lab.py`、`rtx_kv_transfer_lab.py` 的文件哈希均不同，两次采集的差异范围不同：

| 数据 | 采集所用脚本与随包脚本的差异 | 依据 |
|---|---|---|
| RTX 5090 | 采集版本更早；截至 r5 已调整目录空间预检、默认 `large` 教学 Shape 与设备帮助说明；r6 另为 Attention 固定教学输入增加数值验收门 | `measurements.json` 的 `export_note` 与随包脚本的 `numerical_gate` |
| SM90 | 截至 r5 仅 `--allow-non-rtx` 的帮助文本改为通用设备描述；r6 另为 Attention 固定教学输入增加数值验收门 | `measurements_sm90.json` 的 `export_note` 与随包脚本的 `numerical_gate` |

当前随包脚本文件本身未在 GPU 上重新执行。各章引用的耗时与数值对照均以上述历史样本为准；在本机重跑时，以新生成的输出文件及其 `source_sha256` 为准。

r6 的 Attention 数值门要求各必需比较的输出有限、最大绝对误差和 Relative L2 均不超过 0.005，并要求 A1/A2 逐位相同；Flash backend 不可用时记录 `skipped`，不记为技术通过。这些容差只用于脚本的固定 BF16 教学输入，不是其他 Shape、完整模型或生成质量的通用阈值。可用 `python3 -B -m unittest discover -s examples -p 'test_attention_acceptance.py' -v` 在无 GPU 环境验证验收逻辑；该测试不替代 CUDA 重放。

## 固定依赖与离线环境

[锁定依赖](../../requirements/rtx-lab-py310-cu128.lock.txt)、[公开 wheel 清单](../../requirements/rtx-lab-py310-cu128-wheels-public.json)与[下载校验脚本](../../scripts/fetch_rtx_lab_wheels.py)针对 Linux CPython 3.10 + CUDA 12.8 编制，包含全部安装包的大小与 SHA256：

```bash
: "${WHEELHOUSE:?设置为已核验存储上的绝对wheel目录}"
python3 -I -B scripts/fetch_rtx_lab_wheels.py --manifest requirements/rtx-lab-py310-cu128-wheels-public.json --output-dir "$WHEELHOUSE"
python3 -I -B scripts/fetch_rtx_lab_wheels.py --manifest requirements/rtx-lab-py310-cu128-wheels-public.json --output-dir "$WHEELHOUSE" --verify-only
```

离线机器安装时使用 `pip install --no-index --find-links "$WHEELHOUSE" --only-binary=:all: --require-hashes -r requirements/rtx-lab-py310-cu128.lock.txt`。

## Wan2.2：从原生结果建立新参考

在独立工作目录检出[固定官方源码](https://github.com/Wan-Video/Wan2.2/tree/1ea34ff48f87168174e12956e200b1d908b1c5ff)，冻结模型权重与依赖版本后，先使用官方单卡入口跑通基准视频：

```bash
: "${WAN_CKPT:?设置为固定Wan2.2-T2V-A14B权重目录}"
python generate.py --task t2v-A14B --size '1280*720' \
  --ckpt_dir "$WAN_CKPT" --offload_model True --convert_model_dtype \
  --prompt 'A red sailboat moves slowly across a calm lake, with steady camera motion.'
```

确认显存峰值、权重加载与输出视频正常后，冻结帧数、随机种子、采样器时间表、CFG 尺度与计时边界，按第 5 章的 A/B/A 流程采集基准延迟。

## 完整模型验收顺序

将各章优化接入完整模型时，先在小尺寸输入上核验输出有限性（finite）与状态更新，再回到目标分辨率保存原生输出并执行单因素 A/B/A 配对实验；profile 采样与纯净计时分开运行。保持数学等价的系统优化核对逐层误差与状态一致性；改变计算精度或跳步的近似方法按第 5 章联合评估感知画质、运动连贯性与长序列稳定性。

## 配套文件完整性

在解压根目录执行 `python3 -B scripts/verify_reader_package.py .`，即可校验全部随包文件的大小与 SHA256 清单。
