# 第13章 驻留、Offload 与数据搬运

权重、条件 encoder、活跃 KV 和 decoder 的驻留周期不同。先确定哪些对象必须放在设备外、多久搬一次，再决定使用持久 pinned 池或异步预取。下面用单卡合成实验隔离 Host 内存类型与提交方式，再讨论如何组织预取依赖。

## 权重切换与历史读取

Wan2.2 的固定 T2V 入口按噪声阶段选择 expert。两份主干是否同时驻留、切换时何时搬回以及是否清理，应沿实际 Offload 路径记录。单调时间表跨一次 boundary，与逐层频繁卸载的传输次数差别很大。[公开入口](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/text2video.py)

AR 的活跃历史则可能在每次去噪和上下文更新时逐层读回。把已经淘汰的数据存到 CPU，与把下一步还要读取的历史放在 CPU，是两种策略。按每块新增字节、每次读取字节、读取次数和淘汰频率分别记账。KV 容量的通用计算见[第4章](04-memory-and-state.md)，这里只测一批固定数据的搬运。

## 固定载荷的实验设置

[搬运示例](../../examples/rtx_kv_transfer_lab.py)不加载权重，也不执行 Attention。保存数据使用合成 BF16 源张量：10 份 `[1,5040,56,128]` 连续 tensor，共 722,534,400 bytes；一个 epoch 按固定顺序将它们全部复制到预分配 GPU 目标。

输入 seed、内容、dtype、目标地址和 copy 顺序在三种对照中保持相同。pageable 源、持久 pinned 源、GPU 目标各占约 0.673 GiB；完整进程还包含导入、准备和检查的内存。执行前应检查主存、锁页预算和设备余量。

新版脚本默认使用独立的 `large` 教学 Shape；重放上述尺寸时使用 `--fixture explicit-shape --shape 1 5040 56 128`，并固定 tensor 数量为 10。下面是已保存的历史合成观测，采集所用脚本与随包脚本的差异见[附录 B](appendix-b-reproduction.md#历史观测与随包脚本版本)。

## 三个对照各回答一个问题

| 对照 | A1 / A2 | B | 检查什么 |
|---|---|---|---|
| pinning | pageable 源，逐 tensor 阻塞 copy | 预先 pinned 的同内容源，仍阻塞 copy | 排除准备成本后的源内存影响 |
| submission | 持久 pinned 源，逐 tensor 阻塞 copy | 相同源与 Stream，nonblocking 提交、末尾同步 | 逐 tensor 主机等待的成本 |
| pin_each_read_negative | 持久 pinned 源，阻塞 copy | 每次读取先 `pin_memory()`，再阻塞 copy | 在热路径执行 pin 与主存复制的成本 |

所有策略写入相同 GPU 目标。第三组包含 `pin_memory()` 及其主存复制；pinned allocator 可能复用已分配内存。

每臂 5 次 warmup、30 次热观测，一个观测包含全部 10 次 copy，按 A/B/A 顺序运行。Wall time 包含提交、Event 记录与最终同步；CUDA Event 是当前 Stream 的跨度，也可能包含等待 Host 提交或 pin 的空档。预先 pin、GPU 分配、NaN 覆盖及内容检查在热计时外单列。

### 单卡 RTX 5090 结果

环境为 PyTorch 2.8.0+cu128、CUDA 12.8、CPU threads=8；一个 Stream、无并行模型计算。每个 epoch 的载荷为 722,534,400 bytes。

| 对照 | A1 wall ms | B wall ms | A2 wall ms | A/B | A2 相对 A1 漂移 |
|---|---:|---:|---:|---:|---:|
| 预先 pin，仍阻塞 copy | 71.689772 | 13.091678 | 71.245154 | 5.458999× | −0.6202% |
| pinned 源，改 nonblocking 提交 | 13.094814 | 13.020843 | 13.100527 | 1.005900× | +0.0436% |
| 每次读取才 pin，相对持久 pin | 13.101602 | 25.376453 | 13.102055 | 0.516299× | +0.0035% |

按载荷除以 wall time 计算有效带宽，第一组 pageable 取 A1、A2 均值的平均：

$$\mathrm{BW}_{\mathrm{effective}}=
\frac{\mathrm{payload\ bytes}}{T_{\mathrm{wall}}\times10^9}\ \mathrm{GB/s}$$

| 源内存 | 参考 wall（s） | 有效带宽（GB/s） |
|---|---:|---:|
| pageable | 0.071467463 | 10.11 |
| 持久 pinned | 0.013091678 | 55.19 |

这两个数包含相应路径的提交与同步开销，便于比较应用可获得的搬运速率；PCIe 纯 DMA 带宽需另测。第一组主要收益来自事先准备好的 pinned 源。第二组改提交方式只得到约 0.59% 的局部差异；在这一大 tensor、单 Stream 配置下，省去逐次 Host 等待的空间较小。第三组热路径每次临时 pin 的候选有效带宽为 28.47 GB/s，耗时约为持久 pinned 参考的 1.94 倍。

### 数据中心 SM90 GPU 单卡结果

同一搬运协议在单张数据中心 SM90 GPU（80 GB、PyTorch 2.10.0、CUDA runtime 13.1）上的结果如下，使用 `cuda:0`，未运行跨卡传输：

| 对照 | A1 wall ms | B wall ms | A2 wall ms | A/B | A2 相对 A1 漂移 |
|---|---:|---:|---:|---:|---:|
| 预先 pin，仍阻塞 copy | 81.549543 | 13.218359 | 76.447441 | 5.976422× | −6.2564% |
| pinned 源，改 nonblocking 提交 | 13.307552 | 13.103530 | 13.193386 | 1.011214× | −0.8579% |
| 每次读取才 pin，相对持久 pin | 13.215309 | 26.069837 | 13.196618 | 0.506561× | −0.1414% |

按相同载荷和 wall 边界计算，pageable 有效带宽为 9.15 GB/s，持久 pinned 为 54.66 GB/s。热路径反复 pin 的候选为 27.72 GB/s，耗时约为持久 pinned 参考的 1.97 倍；它包含锁页准备与主存复制。在这个固定载荷实验中，预先准备 pinned 源比临时 pin 更快。第一组 pageable 基线的首尾漂移为 −6.26%（A2 略快于 A1），但持久 pinned 相对 pageable 的加速比达 5.98×（耗时从约 79 ms 降至 13.22 ms），效应量远大于基线漂移，结论稳健；全部样本与漂移一起保留在 [measurements_sm90.json](evidence/measurements_sm90.json)。

## 准备成本与数据检查

RTX 5090 记录中，合成 pageable 数据花 7.988707 s，持久 pin 与源复制花 0.444554 s，GPU 目标分配及同步花 0.008139 s。GPU allocation 峰值 758,662,656 B，进程 Host RSS 高水位 3,190,480,896 B；二者都包含相应检查临时量。SM90 记录对应为 4.733067 s、0.624325 s、0.002155 s，GPU allocated 峰值 758,663,168 B、reserved 峰值 773,849,088 B，Host RSS 高水位 3,123,703,808 B。

每臂先将全部目标填 NaN，首次搬运后检查 finite 与 bitwise，热样本结束后再次全量比较。源内容和 version counter 保持，目标地址稳定，目标 version 随 copy 增加。这些检查确认固定载荷按预期覆盖和传输；接入历史 KV 后，还要验证窗口淘汰、位置和下一 chunk 读到的状态。

按第一组 wall 均值差 58.375785 ms，摊销 0.444554 s 的 pin 准备成本：

$$N_{\mathrm{break\ even}}=
\left\lceil\frac{T_{\mathrm{pin\ setup}}}
{T_{\mathrm{pageable}}-T_{\mathrm{pinned}}}\right\rceil=8$$

这是同一固定池重复读取的算术推导。实际 AR 每块还会新增与淘汰历史，需把池写入、提交阶段的 D2H、主存锁页和调度成本一起计入。

## 从持久池到异步预取

如果层 $l+1$ 的历史读取不依赖层 $l$ 的输出，可提前搬运到另一组目标。消费者读取前等待 copy ready，目标复用前等待旧消费者完成；Host 源保留到 DMA 结束。双 buffer 会增加同时存活量，预取距离要与激活显存共同安排。

$$T_{\mathrm{ideal}}\approx\max(T_{\mathrm{compute}},T_{\mathrm{copy}})
+T_{\mathrm{unhidden}}$$
$$\Delta T=T_{\mathrm{original}}-T_{\mathrm{prefetch}}$$

$T_{\mathrm{unhidden}}$ 为无法隐藏的准备与等待，$T_{\mathrm{prefetch}}$ 包含预取及同步。

达到这个理想关系需要独立工作、足够 DMA/计算资源与正确依赖。上述实验只测同一 Stream 上的搬运，因此下一步是集成 copy/compute 两条时间线，测等待是否被隐藏，再回到无插桩整模对照。

实现细节可查 [PyTorch CUDA 语义](https://docs.pytorch.org/docs/2.8/notes/cuda.html#use-pinned-memory-buffers)和[官方 pin/nonblocking 教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)。`non_blocking=True` 表达的是提交行为，重叠是否发生要从实际时间线确认。
