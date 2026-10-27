# DDPM 反事实归因评估报告

## 语义声明

This is a conditional-distribution counterfactual, not a structural causal effect. Path channels (the separate routes that could carry the association) are not identifiable. MMD has no physical unit and is reported as a ratio to the no-intervention baseline. Steps that leave the reference support are excluded from the final attribution. Aggregate scores describe the queried cohort distribution, not a site-level causal share.

- 这是条件分布反事实，不是结构因果效应。
- 路径不可分解：风控、推荐、社交或其他传导通道不能从该分数里分开。
- MMD 无单位，报告相对无干预基线的倍数。
- 支持集外的步骤不进入最终归因。
- 分数描述的是被查询客群的分布对比，不是站点级因果份额。

下面每一张面板都带这句声明。模拟面板的剂量、重叠和隔离是种进去的，用来做相对排序的 ground truth。北京空气质量和美国各州疫情没有反事实 ground truth，那两张面板上的 L1 只是假设检查。

## 先回答业务问题

### synthetic_long

- 目标客群：target。源客群：dose_source。中介：mediator。声明的滞后期：10。
- 归一化分数 Δ^cf / Δ^base = 205.73。原始 MMD^2 = 1.048，95% 区间 [0.975, 1.143].
- 路径形状：后期跳变。支持集通过率：0.93。
- 偏移类型：条件分布有偏移，而且拿掉中介通道后偏移变小。这只说明中介特征参与了预测，不能把路径拆成互斥的因果通道。
- 可信度：低。瓶颈：L3。
- 建议动作：不进入决策。先处理报告里标成瓶颈的那一层，而不是把分数直接交给业务动作。
- 理由：观测校准没有通过。生成模型在观测数据上就不可用，反事实分数不能进入决策。

### synthetic_short

- 目标客群：target。源客群：dose_source。中介：mediator。声明的滞后期：1。
- 归一化分数 Δ^cf / Δ^base = 2.10。原始 MMD^2 = 2.103，95% 区间 [2.092, 2.119].
- 路径形状：峰值后下降。支持集通过率：0.96。
- 偏移类型：条件分布有偏移。中介差值没有单独立住，所以不把这条路径解释成中介传导。
- 可信度：低。瓶颈：L3。
- 建议动作：不进入决策。先处理报告里标成瓶颈的那一层，而不是把分数直接交给业务动作。
- 理由：观测校准没有通过。生成模型在观测数据上就不可用，反事实分数不能进入决策。

### beijing_air

- 目标客群：Dongsi。源客群：Gucheng。中介：Tiantan。声明的滞后期：10。
- 归一化分数 Δ^cf / Δ^base = —。原始 MMD^2 = —，95% 区间 [—, —].
- 路径形状：—。支持集通过率：—。
- 偏移类型：未进入生成模型。MVP 没有给出可排序的信号。
- 可信度：停止。瓶颈：MVP。
- 建议动作：不基于这个查询采取动作。
- 理由：分位数回归的反事实替换没有给出稳定信号，按协议不训练 DDPM。

### us_covid_states

- 目标客群：South。源客群：Northeast。中介：Midwest。声明的滞后期：10。
- 归一化分数 Δ^cf / Δ^base = 1.00。原始 MMD^2 = 0.239，95% 区间 [0.131, 0.360].
- 路径形状：不规则。支持集通过率：0.36。
- 偏移类型：目标分布相对无干预基线没有超过协议里的显著倍数。
- 可信度：低。瓶颈：L3。
- 建议动作：不进入决策。先处理报告里标成瓶颈的那一层，而不是把分数直接交给业务动作。
- 理由：观测校准没有通过。生成模型在观测数据上就不可用，反事实分数不能进入决策。

## 分面板的五层结果

### synthetic_long

Semi-synthetic cohort panel with a lag-10 kernel, a mediator driven by the dose source, planted overlap, and an isolated group.

- MVP：一致率 0.80，Spearman 0.70，主查询倍数 33.93，决定 proceed。
- L1 相对排序：pass
  - dose_response: pass (concordance=0.900, spearman=0.900)
  - temporal_distance: assumption_mismatch (spearman=0.161, smooth=False)
  - overlap: pass (concordance=0.833, spearman=0.800)
  - exogenous_isolation: pass (effect=-2.509, p=0.034)
- L2 时间定位：fail
  - placebo 通过：False，峰值倍数 205.73，placebo 倍数 {'20': 5.48035489077724, '15': 10.538725591345704, '5': 4.594109633058727, '3': 0.21752045469443188}。
  - 精度：声明 lag 10，扫描 argmax 9，通过 False。
  - 门控：期望 coarse_to_fine，通过 True。短尺度均值 0.34，长尺度均值 0.15。
- L3 观测校准：fail
  - PIT p 值 [0.0013591884255364033, 0.007273070588640824, 0.03438004635486791, 0.0036410682405401103]。Wasserstein 0.082（上限 0.095）。HSIC 比值 0.87。
  - Observational calibration is necessary and not sufficient for counterfactual accuracy.
- L4 代理一致性：pass
  - uplift_cate: 一致=True {'ate': 1.9536941927029476, 'p': 2.7798355949981767e-21, 'outcome_shift': 0.9831000984750715}
  - mask_gap: 一致=True {'gap': 1.0390280839426758, 'delta_mask': 0.009150029518747903}
  - business_window: 一致=True {'z_eval': 2.6680978270342064, 'z_placebo': -0.19475756517262655, 'anomalous': True}
  - gate: 一致=True {}
- L5 敏感性：fail
  - 基线 CV 0.01，核宽 CV 0.71，架构 CV 0.23，E-value 19.91。脆弱标记：False。
- 模拟器里的剂量 oracle：Spearman 0.60，一致率 0.70。这组数字只有模拟面板才有，不参与通过与否。
- 训练损失：DDPM 0.965 → 0.515；RealNVP 2.314 → -1.596。

### synthetic_short

Semi-synthetic cohort panel with a lag-1 kernel, a mediator driven by the dose source, planted overlap, and an isolated group.

- MVP：一致率 0.90，Spearman 0.90，主查询倍数 2.10，决定 proceed。
- L1 相对排序：pass
  - dose_response: pass (concordance=0.900, spearman=0.900)
  - temporal_distance: fail (spearman=-0.685, smooth=False)
  - overlap: pass (concordance=0.833, spearman=0.800)
  - exogenous_isolation: pass (effect=-4.763, p=0.003)
- L2 时间定位：pass
  - placebo 通过：True，峰值倍数 2.10，placebo 倍数 {'20': 1.0, '15': 1.2690495254603393, '5': 1.3171646650564435, '3': 1.365315468709464}。
  - 精度：声明 lag 1，扫描 argmax 1，通过 True。
  - 门控：期望 fine_dominant，通过 True。短尺度均值 0.26，长尺度均值 0.15。
- L3 观测校准：fail
  - PIT p 值 [0.19807304068749476, 0.059638622367788385, 5.354135567862302e-05, 0.0773712333478026]。Wasserstein 0.078（上限 0.093）。HSIC 比值 0.86。
  - Observational calibration is necessary and not sufficient for counterfactual accuracy.
- L4 代理一致性：pass
  - uplift_cate: 一致=True {'ate': 2.1663605210386327, 'p': 4.5962176970444317e-29, 'outcome_shift': 2.4385688456669627}
  - mask_gap: 一致=False {'gap': 0.0, 'delta_mask': 2.1030121406542417}
  - business_window: 一致=True {'z_eval': 3.539717610196488, 'z_placebo': -0.0069319670384247545, 'anomalous': True}
  - gate: 一致=True {}
- L5 敏感性：fail
  - 基线 CV 0.01，核宽 CV 0.25，架构 CV 0.00，E-value 32.94。脆弱标记：False。
- 模拟器里的剂量 oracle：Spearman 0.90，一致率 0.90。这组数字只有模拟面板才有，不参与通过与否。
- 训练损失：DDPM 0.965 → 0.500；RealNVP 2.451 → -1.585。

### beijing_air

Beijing multi-site hourly PM2.5, March 2014–January 2017, collapsed to days. Target Dongsi, dose source Gucheng, mediator Tiantan. Within-day 3-hour bins are the cohort distribution. The primary query lag is 10 days, the same lag as the business question; regional pollution usually moves faster, and the protocol is allowed to say so.

停在 stage_1。MVP signal check did not clear the pre-registered bar.

- MVP 一致率 0.30，Spearman -0.60，主查询倍数 0.30，决定：stop。

### us_covid_states

NYT US state COVID panel, 15 Mar 2020–1 Jun 2021. Cohorts are Census regions; the South's states are the outcome distribution; the Northeast's log incidence is the source block. The primary lag is 10 days. Incidence is log1p of new cases per 100k.

- MVP：一致率 0.50，Spearman -0.10，主查询倍数 1.33，决定 proceed_unstable_ranking。
- L1 相对排序：assumption_check
  - dose_response: assumption_rejected (concordance=0.400, spearman=-0.500)
  - temporal_distance: assumption_rejected (spearman=-0.071, smooth=False)
  - overlap: insufficient_contrast ()
  - exogenous_isolation: insufficient_contrast ()
- L2 时间定位：fail
  - placebo 通过：False，峰值倍数 1.00，placebo 倍数 {'20': 0.5164530202737608, '15': 1.1158946236655776, '5': 0.9561258146236605, '3': 1.2934525753787032}。
  - 精度：声明 lag 10，扫描 argmax 1，通过 False。
  - 门控：期望 coarse_to_fine，通过 True。短尺度均值 0.70，长尺度均值 0.52。
- L3 观测校准：fail
  - PIT p 值 [1.6074265181600106e-57, 4.6178369314920895e-27, 3.866789460449582e-07, 2.0313744994041992e-29]。Wasserstein 0.572（上限 0.182）。HSIC 比值 0.99。
  - Observational calibration is necessary and not sufficient for counterfactual accuracy.
- L4 代理一致性：uncertain
  - uplift_cate: 一致=False {'ate': 0.11284668563404361, 'p': 0.028711800132782562, 'outcome_shift': 0.6147258733155705}
  - mask_gap: 一致=False {'gap': 0.0, 'delta_mask': 0.23865308393298984}
  - business_window: 一致=False {'z_eval': 2.4929384528390672, 'z_placebo': -0.0008847925242099606, 'anomalous': True}
  - gate: 一致=True {}
- L5 敏感性：fail
  - 基线 CV 0.01，核宽 CV 0.99，架构 CV 0.08，E-value 2.71。脆弱标记：False。
- 训练损失：DDPM 1.341 → 0.446；RealNVP 4.560 → -4.908。

## 协议里预先写死、没有事后改动的判定

- L1：配对一致率 > 0.8 且 Spearman > 0.7；时间距离 Spearman < -0.7 且升降次数不超过长度的五分之一；隔离组效应量 < -1 且 Welch p < 0.05。有 ground truth 的面板至少三组构造通过。失败时，只有预先登记为假设不匹配的构造（长记忆面板的单调时间距离）记成假设问题，其余记成方法没有通过。
- L2：声明 lag 的归一化分数 > 2，且至少三个 placebo lag 都不显著；声明 lag 的分数不低于扫描最大值的 5%（随机傅里叶特征的预留误差），并且不低于相邻 lag。长滞后要求门控先长后短，短滞后要求短尺度权重大于长尺度。
- L3：每个状态维度的 PIT，KS p > 0.05；观测 Wasserstein < 0.1 倍标准差；HSIC 比值在 [0.8, 1.2]。观测校准好是必要不充分条件。
- L4：uplift、中介 mask、业务窗口、门控四项里至少三项与 rollout 一致。两项及以上不一致则降为不确定。门控同时出现在 L2 和 L4，这两层不是独立证据。
- L5：基线、核宽、架构的变异系数分别低于 30%、20%、30%，E-value > 2。E-value < 1.5 另标脆弱。
- 归一化分数的分母是无干预预测的 MMD 和样本内噪声地板中的较大者，避免基线接近 0 时倍数爆炸。Bootstrap 200 次，重抽成员和条件样本，不重新训练网络。MMD 的核宽是训练期结果变量的边际中位距离，不把反事实样本放进核宽估计，否则大的偏移会把核宽撑大、分数反而变小。

## 方法附录

反事实路径用残差 rollout：干预日只替换源块的第 0 维为参照水平，之后每个时点的反事实状态等于观测状态加上「反事实条件下的生成均值减观测条件下的生成均值」。同一随机种子同时用于两条均值，所以没有被干预解释的冲击留在观测路径里。支持集用训练期的状态范围，异常日也算在支持里；参照分布只用来选干预水平。轻度出支持的状态做重加权或截断，严重出支持的步回退到观测值，不进入归因。

主模型是条件 DDPM，去噪网络的三个历史池由 MultiScaleGate 缩放。架构对照是条件 RealNVP，不是第二次校准机会。结果模型是 0.1 / 0.5 / 0.9 分位数的直方图梯度提升。点估计用条件中位数拼出客群截面，抽样只用于 bootstrap 区间。MVP 在进入扩散模型之前，用训练窗上的 AR(1) 把干预日的替换传到后续滞后；只改一天会把持续性回声留在其他滞后里，排序检验会失真。

```
\paragraph{DDPM counterfactual attribution.}
The deployment is a conditional-distribution contrast. A quantile-regression
MVP decides whether the query has a rankable signal. The conditional DDPM
then rolls the intervened block forward by a residual update that holds
unexplained shocks fixed, with a support check on every generated step.
MMD is reported as a ratio to the no-intervention predictive baseline.
Evaluation is the five-layer falsification protocol, and the weakest
failed layer is the credibility bottleneck. Observational panels do not
confirm L1, because a counterfactual ground truth was not constructed there.
```
