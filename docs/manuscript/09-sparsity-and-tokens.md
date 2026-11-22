# 第9章 稀疏、Token 压缩与替代 Attention

稀疏和 Token 压缩减少的工作不同：前者通常删去部分交互，后者还可能减少后续投影和 MLP 的 token 数。替代 Attention 进一步改变计算形式。选择时应看完整 block 节省了什么、额外做了什么，以及模型是否为该路径训练。

## 结构化稀疏与近似注意力

### Sliding Tile Attention：让局部窗口对齐计算 Tile

逐 Token 滑窗在二维或三维展平后，往往产生既有有效位置又有被遮蔽位置的 mixed tile。Kernel 仍要计算这些 tile，再对部分元素施加 Mask，位置稀疏率就难以转成等比例计算节省。STA 把视频重新排列成连续的时空 tile，让同一个 query tile 共享局部 key 窗口；窗口内按完整 dense tile 计算，窗口外整块跳过。这样局部性和 Tensor Core 的计算单位能够对齐。[STA 论文§2–3](https://arxiv.org/html/2502.04507v1)

窗口大小决定质量与工作量。论文分别讨论 training-free 的 Mask 搜索和微调配置；部署时将窗口、head/layer 配置和对应权重一起保存。更大的空间运动或跨场景依赖可能需要窗口外信息，移植到矩形 AR Attention 时还要处理历史边界。

### Sparse VideoGen：按 head 选择空间或时间模式

SVG 观察到一些 head 偏重同帧空间区域，另一些偏重跨帧时间关联。它用少量 query 的在线采样比较候选稀疏模式与 dense 输出，选择误差较小的模式，再以相应布局和 Kernel 执行。空间、时间访问在原始展平布局中具有不同 stride，重排的作用是让保留的 KV 尽量连续，减少低效访存。[SVG 论文§4](https://arxiv.org/html/2502.01776v1)、[官方实现](https://github.com/svg-project/Sparse-VideoGen)

模式识别、采样参考、重排和还原都是实际成本。序列较短时，这些逐步开销更难由 Attention 计算节省覆盖。减少 NFE 后，编译和初始化等一次性成本的占比可能上升；逐步开销则按实际调用次数统计。SVG 的 training-free 路径、STA 的窗口设计与下面需要匹配训练的 VSA 是不同路线，选择时先确认自己的质量与训练预算。

### Video Sparse Attention：粗粒度路由与细粒度计算

VSA 按内容选择 block，而不是固定保留空间邻居或运动轨迹。[原论文§2.2 与 Appendix A](https://arxiv.org/html/2505.13389v1#S2.SS2)把视频 token 按时空 cube 重排，默认一个 $4\times4\times4$ cube 包含 64 个 token。query cube 与 key cube 形成的 score tile 才是 $64\times64$；两种尺寸不能混用。

粗阶段对每块 Q/K/V 取均值，计算 cube 之间的 attention，并逐行 Top-K 选出细阶段要计算的块。细阶段在选中块内部执行 token-level attention。粗输出也参与最终结果，不是仅生成 mask：

$$Q_c,K_c,V_c=\operatorname{MeanPoolTiles}(Q,K,V)$$
$$A_c=\operatorname{softmax}(Q_cK_c^\top/\sqrt d)$$
$$m=\operatorname{ExpandTileMask}(\operatorname{RowTopK}(A_c))$$
$$O_f=\operatorname{softmax}(QK^\top/\sqrt d+\operatorname{mask}(m))V$$
$$O_c=\operatorname{BroadcastTiles}(A_cV_c),\qquad
O=G_c\odot O_c+G_f\odot O_f$$

$\operatorname{mask}(m)$ 将保留连接映射为 0，其余连接映射为 $-\infty$。

门控与主干需要匹配训练；适配预训练模型时，论文将 fine gate 固定为 1，coarse gate 从 0 开始学习。时空重排让一个 cube 的 token 连续存放，细阶段可以按整块消费，避免逐 token 不规则 gather；粗路由、Top-K、重排与融合仍有成本，稀疏率不能直接换算成整层加速。论文中的训练收益和配套权重也不能替代在 AR-chunk 矩形 attention、相机位置协议上的适配验证。

### 稀疏与线性混合注意力：SLA (Sparse-Linear Attention)
SLA 用稀疏 softmax 与经过训练的线性分支互补。[论文 v2 Eq.5–6](https://arxiv.org/html/2509.24006v2#S4)的输出还包含可学习投影：
$$Y_{\text{SLA}}=Y_{\text{sparse}}+\operatorname{Proj}(Y_{\text{linear}})$$
论文用压缩 mask 聚合 marginal blocks；[SLA 固定源码](https://github.com/thu-ml/SLA/blob/7db4039111a9002f900be537c2b2061d26d73744/sparse_linear_attention/core.py)则先对全 KV 计算线性分支，再投影相加。两种版本的线性分支都依赖匹配训练、配套权重和原生位置/条件协议。

## 稀疏率之外的成本与误差

动态路由、Top-K 和 packing 可能抵消少做的 Attention 乘法。逐 Token 索引容易产生不规则 gather，整块选择则更便于保持连续读取。记录 `router → packing → attention → restore` 的完整时间，同时保留各阶段 trace，才能判断改进应落在哪一处。

被删位置的概率质量比位置比例更接近误差来源。若被删位置的累积注意力概率质量为 $p_{\mathrm{drop}}\in[0,1)$，保留和删除部分各自归一化后的输出为 $O_{\mathrm{keep}}$、$O_{\mathrm{drop}}$，则：

$$O_{\mathrm{dense}}=(1-p_{\mathrm{drop}})O_{\mathrm{keep}}+p_{\mathrm{drop}}O_{\mathrm{drop}}$$
$$O_{\mathrm{keep}}-O_{\mathrm{dense}}=p_{\mathrm{drop}}(O_{\mathrm{keep}}-O_{\mathrm{drop}})$$

同样删除 90% 的位置，$p_{\mathrm{drop}}$ 可以很小，也可以很大。路由需要找到对当前 query 有用的连接；剩余连接重新归一化后，误差还会受 V 的幅值与方向影响。先检查 Mask、位置和历史可见性，再按第 5 章的近似方法路径评估视频质量。

## Token pruning 与 merging 改变整个子图的几何

设原序列长度为 $N$、处理后的有效长度为 $rN$，$0<r\leq1$。若压缩发生在整个 block 之前、且后续都使用缩短序列，理想 GEMM 主项随 $r$ 缩小，dense attention 主项随 $r^2$ 缩小；若只在 Attention Core 内压缩 K/V，QKV 投影或 MLP 可能完全未减少。需要明确插入位置，才能解释 token 保留率带来了多少计算节省。

merging 还要计算相似度、选配对、聚合和 unmerge；pruning 要定义删去位置如何影响后续空间重建。保留 Shape 恢复路径不表示内容精确恢复，跨模态和运动边界也可能被错误合并。[ToMe for Stable Diffusion](https://github.com/dbolya/tomesd)是可查的图像扩散示例，不是现成的 Wan2.2 或 SolarWM 视频适配。

$$T_{\mathrm{candidate}}=T_{\mathrm{select}}+T_{\mathrm{pack}}
+T_{\mathrm{reduced}}+T_{\mathrm{restore}}$$
$$\Delta T=T_{\mathrm{original}}-T_{\mathrm{candidate}}$$

只有 $\Delta T>0$ 且质量通过才有采用依据。上述 $r$ 和 $r^2$ 描述理想计算量，延迟还包含选择、重排与恢复的成本。

## 双向与 AR 的验证差异

Wan2.2 需要检查时空邻域、运动位置和 cross-attention 条件保持；两份 expert 可能需要分别校准。一个匹配某个 expert 的 mask 或门控权重不能直接用于另一份网络。若稀疏只改 Attention Core，量化、QKV、MLP 及 expert 切换仍按原路径计时。

连续 AR 还需要保证当前 query 可见正确历史、局部位置和相机对应一致。动态选择不能从未来块读取信息，也不能把物理 ring 槽号当时间位置。逐块差异可能经过未来生成放大，因此短 chunk 数值接近只是一项筛选，不能替代多次淘汰后的质量。

路由、packing、转换与 attention 分开诊断后，回到无插桩完整路径计时。若候选变慢，保留同 dtype/backend 的分因子对照，定位增加的成本来自路由、布局还是主 Kernel。
