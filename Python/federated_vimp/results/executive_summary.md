Conditional association only. D_agg is a fleet label, not a site share. vimp_cov and vimp_con are not merged.

先处理的站点（gated primary driver）
- site-00: 机制=concept，z=37.71，强度 Δ=33940，主导特征指数=0。动作：触发该站局部重训。概念漂移是 P(Y|X) 变了，上游管道通常是干净的。
- site-10: 机制=covariate，z=5.81，强度 Δ=5109，主导特征指数=3。动作：检查该站上游数据管道。协变量漂移是 P(X) 变了，先不要全网重训。

不要当成机群主驱动
- 局部异常（z 高，但强度不够或主导特征不在共识集）: site-18。动作：本站监控，不要全网重训。
- 共识破坏者（贡献主要落在非共识特征）: site-07, site-17。动作：查 split 轴和特征定义，不要用它的 VIMP 代表机群。

共识特征指数 concept=[0, 1]，covariate=[2, 3]。D_agg=0.093（机群标签，不是份额）。
漂移类型计数（由 D_k 符号，不是由真值）: concept=10, covariate=10, joint(z_mmd>2 且 z_r>2)=0。
警报：n-加权均值和中位数的 Spearman < 0.8，少数站点在拉均值。以中位数和强度加权为准，不要只报 n-加权。
警报：机制组内的平均 Spearman < 0.5。
