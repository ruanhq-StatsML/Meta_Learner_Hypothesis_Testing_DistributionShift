# 语义声明（阶段 0）

所有输出都是 **conditional association**：在“历史窗口 vs 当前窗口”这个对比下，某个特征指数和批次标签、或和批次间结果差，有多强的共变。它们不是因果效应，不能写成“特征 j 造成了站点 k 的漂移”，也不能把站点排序写成对总效应的因果分解。

聚合 dominance score \(D_{\mathrm{agg}}\) 是 **机群层面的标签**（这批站点整体更像 covariate、更像 concept，还是两边都有）。它不是某个 site 的份额，也不能从 \(D_{\mathrm{agg}}\approx 0\) 推出“没有站点在漂移”。混合机群里一半 concept、一半 covariate 时，\(D_{\mathrm{agg}}\) 可以接近 0，同时两边的组内 VIMP 都很尖。

四个操作量不合成一个标量，分两组证据：

| 组 | 上传字段 | 成员 | 回答的问题 |
|---|---|---|---|
| Covariate | `vimp_cov` | MMD-LOCO、HSIC-LOCO、domain permutation VIMP | 哪些特征指数在搬 \(P(X)\) |
| Concept | `vimp_con` | PO-risk LOCO、R-risk LOCO、DR-split VIMP | 哪些特征指数在搬 \(P(Y\mid X)\) |
| 强度 | `z_mmd`, `z_r` | 相对本站历史的 MAD z-score | 这站当前窗口离自己的平稳历史有多远 |
| 机制标签 | `D_k` | 只比较两组峰值的浓度 | 这站自己更像哪一种漂移 |

`z` 在 **统一的分析样本量** \(n_{\mathrm{analysis}}=\min(n_k, n_{\mathrm{cap}})\) 上计算。它衡量的是可比较的偏移幅度，不是“样本越大越显著”的检验统计量。\(n_k\) 只在服务器上当作影响权重。否则最大的数据中心会只因为 \(\sqrt{n}\) 变成主驱动。

`z_mmd` 用 \(\log(\mathrm{MMD})\)，`z_r` 用 \(\mathrm{asinh}(R\text{-risk})\)。原始 MMD² 右偏，在平稳历史下 placebo z 的标准差也会明显高于 1。历史窗口只有 24 个时，标准差这个估计自己的标准误大约是 0.15，所以 \(|\mathrm{std}-1|\le 0.3\) 会把平稳站点标成失败。实验里排序用的门是 0.80，0.30 的失败率单独报告。上传字段 `calibration_ok` 仍是两个检验的合取。服务器只把**该机制自己的**检验失败当成 z 不可用：concept 站点看 R-risk 的校准，covariate 站点看 MMD 的校准。

服务器不得把 `vimp_cov` 和 `vimp_con` 加总后再排序。特征共识用组内聚合；站点主驱动用 \((z_k, |\Delta_k|)\) 加共识特征门控。见实验笔记里对 literal LOSO 和 gated router 的对照。

不上传原始行、模型参数、特征名。通信量是每个站点 \(O(p)\)。

英文稿（可直接放进方法段）：

\paragraph{Semantic declaration.}
Every released score is a conditional association between a feature index and the batch contrast. None is a causal effect, and none is a decomposition of a fleet-level effect into site shares. The aggregate dominance score is an aggregate-level label: a value near zero means the fleet is balanced across mechanisms, not that no site drifted. Covariate evidence (MMD-LOCO, HSIC-LOCO, domain permutation VIMP) and concept evidence (PO-risk LOCO, R-risk LOCO, DR-split VIMP) are simplex-normalized and averaged inside their own group, then kept as two vectors. Per-site \(z\)-scores are MAD scores against that site's own stationary history, computed at a common analysis size so that sample size does not enter detectability; \(n_k\) is used only as an influence weight at the server. Sites with \(n_k<200\) or a failed placebo calibration are flagged and left out of ranking.

本实验里 DR-split VIMP 是双重稳健分数上的浅树分裂重要度，接口与 GRF split VIMP 相同，生产环境可换成 `grf`。深度诊断（CFPerm 分支、FSDS top-K）不在本实验的计算路径里：本实验只检验“主驱动站点有没有被指对”。
