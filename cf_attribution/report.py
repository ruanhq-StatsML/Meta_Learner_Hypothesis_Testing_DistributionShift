"""Markdown report and the two-axis diagnostic figure for one run."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cf_attribution.semantics import SEMANTIC_BULLETS_ZH, SEMANTIC_DECLARATION


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(type(obj))


def dump_json(payload, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _fmt(value, digits=3):
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _association(result: dict) -> str:
    if result.get("stopped"):
        return "未进入生成模型。MVP 没有给出可排序的信号。"
    ratio = result.get("delta_normalized") or 0.0
    proxies = result.get("evaluation", {}).get("L4", {}).get("proxies", [])
    mask = next((p for p in proxies if p.get("name") == "mask_gap"), {})
    if ratio <= 2:
        return "目标分布相对无干预基线没有超过协议里的显著倍数。"
    if mask.get("agree"):
        return "条件分布有偏移，而且拿掉中介通道后偏移变小。这只说明中介特征参与了预测，不能把路径拆成互斥的因果通道。"
    return "条件分布有偏移。中介差值没有单独立住，所以不把这条路径解释成中介传导。"


def _action(result: dict) -> str:
    level = result.get("credibility", {}).get("level", "")
    ratio = result.get("delta_normalized") or 0.0
    if result.get("stopped") or level == "停止":
        return "不基于这个查询采取动作。"
    if level in {"高", "中等偏高"} and ratio and ratio > 2:
        return (
            "可以把源客群在干预日的暴露交给业务方做排查，并在随后 3 到 5 个窗口里看目标客群的分布是否回到基线。"
            "排查的依据是条件分布对比，不是已经证实的结构因果效应。"
        )
    if level == "探索性":
        return "只写入探索记录。不隔离客群，不改策略。若要做决策，需要外生实验把最弱的那一层补上。"
    return "不进入决策。先处理报告里标成瓶颈的那一层，而不是把分数直接交给业务动作。"


def _layer_line(name: str, block: dict) -> str:
    status = block.get("status", "—")
    return f"- {name}：{status}"


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# DDPM 反事实归因评估报告",
        "",
        "## 语义声明",
        "",
        SEMANTIC_DECLARATION,
        "",
    ]
    for bullet in SEMANTIC_BULLETS_ZH:
        lines.append(f"- {bullet}")
    lines += [
        "",
        "下面每一张面板都带这句声明。模拟面板的剂量、重叠和隔离是种进去的，用来做相对排序的 ground truth。北京空气质量和美国各州疫情没有反事实 ground truth，那两张面板上的 L1 只是假设检查。",
        "",
        "## 先回答业务问题",
        "",
    ]
    for result in results:
        lines += [
            f"### {result['panel']}",
            "",
            f"- 目标客群：{result.get('target', '—')}。源客群：{result.get('source', '—')}。中介：{result.get('mediator', '—')}。声明的滞后期：{result.get('peak_lag', '—')}。",
            f"- 归一化分数 Δ^cf / Δ^base = {_fmt(result.get('delta_normalized'), 2)}。原始 MMD^2 = {_fmt(result.get('delta_cf'))}，95% 区间 [{_fmt((result.get('delta_cf_ci') or [None, None])[0])}, {_fmt((result.get('delta_cf_ci') or [None, None])[1])}].",
            f"- 路径形状：{result.get('path_shape', '—')}。支持集通过率：{_fmt(result.get('support_pass_rate'), 2)}。",
            f"- 偏移类型：{_association(result)}",
            f"- 可信度：{result.get('credibility', {}).get('level', '—')}。瓶颈：{result.get('credibility', {}).get('bottleneck', '—')}。",
            f"- 建议动作：{_action(result)}",
            f"- 理由：{result.get('credibility', {}).get('reason', '')}",
            "",
        ]
    lines += ["## 分面板的五层结果", ""]
    for result in results:
        ev = result.get("evaluation") or {}
        lines += [f"### {result['panel']}", "", result.get("notes", ""), ""]
        if result.get("stopped"):
            lines += [f"停在 {ev.get('stopped_at', 'stage 1')}。{ev.get('reason', '')}", ""]
            mvp = result.get("mvp") or {}
            lines += [
                f"- MVP 一致率 {_fmt(mvp.get('concordance'), 2)}，Spearman {_fmt(mvp.get('spearman'), 2)}，主查询倍数 {_fmt(mvp.get('main_ratio'), 2)}，决定：{mvp.get('decision')}。",
                "",
            ]
            continue
        mvp = result.get("mvp") or {}
        lines += [
            f"- MVP：一致率 {_fmt(mvp.get('concordance'), 2)}，Spearman {_fmt(mvp.get('spearman'), 2)}，主查询倍数 {_fmt(mvp.get('main_ratio'), 2)}，决定 {mvp.get('decision')}。",
            _layer_line("L1 相对排序", ev.get("L1", {})),
        ]
        for cons in ev.get("L1", {}).get("constructions", []):
            extra = []
            for key in ("concordance", "spearman", "effect", "p", "smooth"):
                if key in cons:
                    extra.append(f"{key}={_fmt(cons[key], 3) if not isinstance(cons[key], bool) else cons[key]}")
            lines.append(f"  - {cons.get('name')}: {cons.get('status')} ({', '.join(extra)})")
        l2 = ev.get("L2", {})
        lines += [
            _layer_line("L2 时间定位", l2),
            f"  - placebo 通过：{l2.get('placebo_pass')}，峰值倍数 {_fmt(l2.get('peak_ratio'), 2)}，placebo 倍数 {l2.get('placebo_ratios')}。",
            f"  - 精度：声明 lag {l2.get('precision', {}).get('true_lag')}，扫描 argmax {l2.get('precision', {}).get('argmax_lag')}，通过 {l2.get('precision', {}).get('pass')}。",
            f"  - 门控：期望 {l2.get('gate', {}).get('expect')}，通过 {l2.get('gate', {}).get('pass')}。短尺度均值 {_fmt(l2.get('gate', {}).get('short_mean'), 2)}，长尺度均值 {_fmt(l2.get('gate', {}).get('long_mean'), 2)}。",
        ]
        l3 = ev.get("L3", {})
        lines += [
            _layer_line("L3 观测校准", l3),
            f"  - PIT p 值 {l3.get('pit_p')}。Wasserstein {_fmt(l3.get('wasserstein'))}（上限 {_fmt(l3.get('wasserstein_limit'))}）。HSIC 比值 {_fmt(l3.get('hsic_ratio'), 2)}。",
            f"  - {l3.get('caveat', '')}",
        ]
        l4 = ev.get("L4", {})
        lines.append(_layer_line("L4 代理一致性", l4))
        for proxy in l4.get("proxies", []):
            lines.append(f"  - {proxy.get('name')}: 一致={proxy.get('agree')} { {k: proxy[k] for k in proxy if k not in {'name', 'agree'}} }")
        l5 = ev.get("L5", {})
        lines += [
            _layer_line("L5 敏感性", l5),
            f"  - 基线 CV {_fmt(l5.get('baseline_cv'), 2)}，核宽 CV {_fmt(l5.get('kernel_cv'), 2)}，架构 CV {_fmt(l5.get('architecture_cv'), 2)}，E-value {_fmt(l5.get('evalue'), 2)}。脆弱标记：{l5.get('fragile')}。",
        ]
        if result.get("oracle_dose"):
            lines.append(
                f"- 模拟器里的剂量 oracle：Spearman {_fmt(result['oracle_dose'].get('spearman'), 2)}，"
                f"一致率 {_fmt(result['oracle_dose'].get('concordance'), 2)}。这组数字只有模拟面板才有，不参与通过与否。"
            )
        train = result.get("training") or {}
        if train.get("ddpm_loss"):
            lines.append(
                f"- 训练损失：DDPM {_fmt(train['ddpm_loss'][0], 3)} → {_fmt(train['ddpm_loss'][-1], 3)}；"
                f"RealNVP {_fmt(train['flow_loss'][0], 3)} → {_fmt(train['flow_loss'][-1], 3)}。"
            )
        lines.append("")
    lines += [
        "## 协议里预先写死、没有事后改动的判定",
        "",
        "- L1：配对一致率 > 0.8 且 Spearman > 0.7；时间距离 Spearman < -0.7 且升降次数不超过长度的五分之一；隔离组效应量 < -1 且 Welch p < 0.05。有 ground truth 的面板至少三组构造通过。失败时，只有预先登记为假设不匹配的构造（长记忆面板的单调时间距离）记成假设问题，其余记成方法没有通过。",
        "- L2：声明 lag 的归一化分数 > 2，且至少三个 placebo lag 都不显著；声明 lag 的分数不低于扫描最大值的 5%（随机傅里叶特征的预留误差），并且不低于相邻 lag。长滞后要求门控先长后短，短滞后要求短尺度权重大于长尺度。",
        "- L3：每个状态维度的 PIT，KS p > 0.05；观测 Wasserstein < 0.1 倍标准差；HSIC 比值在 [0.8, 1.2]。观测校准好是必要不充分条件。",
        "- L4：uplift、中介 mask、业务窗口、门控四项里至少三项与 rollout 一致。两项及以上不一致则降为不确定。门控同时出现在 L2 和 L4，这两层不是独立证据。",
        "- L5：基线、核宽、架构的变异系数分别低于 30%、20%、30%，E-value > 2。E-value < 1.5 另标脆弱。",
        "- 归一化分数的分母是无干预预测的 MMD 和样本内噪声地板中的较大者，避免基线接近 0 时倍数爆炸。Bootstrap 200 次，重抽成员和条件样本，不重新训练网络。MMD 的核宽是训练期结果变量的边际中位距离，不把反事实样本放进核宽估计，否则大的偏移会把核宽撑大、分数反而变小。",
        "",
        "## 方法附录",
        "",
        "反事实路径用残差 rollout：干预日只替换源块的第 0 维为参照水平，之后每个时点的反事实状态等于观测状态加上「反事实条件下的生成均值减观测条件下的生成均值」。同一随机种子同时用于两条均值，所以没有被干预解释的冲击留在观测路径里。支持集用训练期的状态范围，异常日也算在支持里；参照分布只用来选干预水平。轻度出支持的状态做重加权或截断，严重出支持的步回退到观测值，不进入归因。",
        "",
        "主模型是条件 DDPM，去噪网络的三个历史池由 MultiScaleGate 缩放。架构对照是条件 RealNVP，不是第二次校准机会。结果模型是 0.1 / 0.5 / 0.9 分位数的直方图梯度提升。点估计用条件中位数拼出客群截面，抽样只用于 bootstrap 区间。MVP 在进入扩散模型之前，用训练窗上的 AR(1) 把干预日的替换传到后续滞后；只改一天会把持续性回声留在其他滞后里，排序检验会失真。",
        "",
        "```",
        "\\paragraph{DDPM counterfactual attribution.}",
        "The deployment is a conditional-distribution contrast. A quantile-regression",
        "MVP decides whether the query has a rankable signal. The conditional DDPM",
        "then rolls the intervened block forward by a residual update that holds",
        "unexplained shocks fixed, with a support check on every generated step.",
        "MMD is reported as a ratio to the no-intervention predictive baseline.",
        "Evaluation is the five-layer falsification protocol, and the weakest",
        "failed layer is the credibility bottleneck. Observational panels do not",
        "confirm L1, because a counterfactual ground truth was not constructed there.",
        "```",
        "",
    ]
    return "\n".join(lines)


def plot_panel(result: dict, path: Path) -> None:
    if result.get("stopped"):
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    name = result["panel"]
    ratio = result.get("path_ratio") or []
    ax = axes[0, 0]
    if ratio:
        ax.plot(np.arange(1, len(ratio) + 1), ratio, color="#0f6f8a", marker="o", ms=3.5, label="normalized MMD")
        ax.axhline(2.0, color="#8a3d3d", ls="--", lw=0.8, label="significance cut 2")
    ax.set_title(f"{name}: path of normalized MMD")
    ax.set_xlabel("Horizon after the intervention (days)")
    ax.set_ylabel("MMD^2 / no-intervention MMD^2")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    gates = np.asarray(result.get("gate_trajectory") or [], dtype=float)
    if gates.size:
        ax.plot(gates[:, 0], label="short pool (2d)", color="#0f6f8a")
        ax.plot(gates[:, 1], label="medium pool (7d)", color="#c47b2b")
        ax.plot(gates[:, 2], label="long pool (14d)", color="#3d5a3d")
    ax.set_title(f"{name}: multiscale gate along the rollout")
    ax.set_xlabel("Generated step")
    ax.set_ylabel("Gate weight")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    scan = (result.get("evaluation") or {}).get("L2", {}).get("scan_ratios") or []
    if scan:
        xs = np.arange(1, len(scan) + 1)
        ys = [np.nan if v is None else v for v in scan]
        ax.plot(xs, ys, color="#0f6f8a", marker="o", ms=3.5)
        peak = result.get("peak_lag")
        if peak:
            ax.axvline(peak, color="#8a3d3d", ls="--", lw=0.8, label=f"declared lag {peak}")
            ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"{name}: normalized MMD by hypothesized lag")
    ax.set_xlabel("Hypothesized lag (days before the evaluation day)")
    ax.set_ylabel("Normalized MMD^2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    oracle = result.get("oracle_dose")
    if oracle and oracle.get("deltas"):
        ax.scatter(oracle["doses"], oracle["deltas"], color="#0f6f8a")
        ax.set_title(f"{name}: oracle MMD against planted dose")
        ax.set_xlabel("Planted impulse size")
        ax.set_ylabel("Oracle MMD^2 (standardized outcome)")
    else:
        dose = None
        for cons in (result.get("evaluation") or {}).get("L1", {}).get("constructions", []):
            if cons.get("name") == "dose_response":
                dose = cons
        ax.set_title(f"{name}: dose-response construction")
        ax.set_xlabel("Construction")
        ax.set_ylabel("Status code")
        ax.text(0.05, 0.6, f"status: {None if dose is None else dose.get('status')}\nconcordance: {_fmt(None if dose is None else dose.get('concordance'), 2)}", transform=ax.transAxes, fontsize=9)
        ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(results: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(results, out_dir / "results.json")
    text = render_markdown(results)
    path = out_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    for result in results:
        plot_panel(result, out_dir / f"{result['panel']}.png")
    return path
