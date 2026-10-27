"""Conditional DDPM, conditional RealNVP, and quantile outcome model."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cf_attribution.nets import MLP, sigmoid, time_embedding


class ConditionalDDPM:
    """Epsilon-prediction DDPM. MultiScaleGate rescales three history pools."""

    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        pool_slices: list[tuple[int, int]],
        hidden: int = 96,
        n_steps: int = 16,
        lr: float = 1e-3,
        seed: int = 2026,
    ):
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.pool_slices = pool_slices
        self.n_steps = n_steps
        self.t_dim = 16
        betas = np.linspace(1e-4, 0.18, n_steps)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = np.cumprod(self.alphas)
        self.gate = MLP([cond_dim, hidden, 3], seed=seed, lr=lr)
        self.net = MLP(
            [x_dim + self.t_dim + cond_dim, hidden, hidden, x_dim],
            seed=seed + 1,
            lr=lr,
        )
        self.last_gate = np.full(3, 1.0 / 3.0)
        self.loss_curve: list[float] = []

    def _eps(self, x: np.ndarray, t_idx: np.ndarray, cond: np.ndarray):
        g = sigmoid(self.gate.forward(cond))
        gated = cond.copy()
        for i, (a, b) in enumerate(self.pool_slices):
            gated[:, a:b] *= g[:, i : i + 1]
        temb = time_embedding(t_idx.astype(float), self.t_dim)
        pred = self.net.forward(np.concatenate([x, temb, gated], axis=1))
        return pred, g, gated

    def fit(self, x: np.ndarray, cond: np.ndarray, epochs: int, batch: int, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        n = len(x)
        losses = []
        for _ in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            seen = 0
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                if len(idx) < 8:
                    continue
                xb = x[idx]
                cb = cond[idx].copy()
                drop = rng.random(len(idx)) < 0.10
                cb[drop] = 0.0
                t_idx = rng.integers(0, self.n_steps, size=len(idx))
                noise = rng.normal(size=xb.shape)
                a_bar = self.alpha_bar[t_idx][:, None]
                noisy = np.sqrt(a_bar) * xb + np.sqrt(1.0 - a_bar) * noise
                pred, g, gated = self._eps(noisy, t_idx, cb)
                # Mean squared error. grad is d(mean)/d(pred).
                resid = pred - noise
                grad = (2.0 / len(idx)) * resid
                g_in = self.net.backward(grad)
                d_gated = g_in[:, self.x_dim + self.t_dim :]
                d_g = np.zeros_like(g)
                for i, (a, b) in enumerate(self.pool_slices):
                    d_g[:, i] = np.sum(d_gated[:, a:b] * cb[:, a:b], axis=1)
                d_pre = d_g * g * (1.0 - g)
                self.gate.backward(d_pre)
                self.net.step()
                self.gate.step()
                total += float(np.mean(resid ** 2)) * len(idx)
                seen += len(idx)
            losses.append(total / max(seen, 1))
        self.loss_curve = losses
        return losses

    def ancestral(self, cond: np.ndarray, rng: np.random.Generator, eta: float = 0.0):
        """DDIM-style sampler. eta=0 is the deterministic mean used in rollouts."""
        n = len(cond)
        x = rng.normal(size=(n, self.x_dim))
        gate = np.full((n, 3), 1.0 / 3.0)
        for t in reversed(range(self.n_steps)):
            t_idx = np.full(n, t, dtype=int)
            eps, gate, _ = self._eps(x, t_idx, cond)
            a_bar = float(self.alpha_bar[t])
            x0 = (x - np.sqrt(max(1.0 - a_bar, 0.0)) * eps) / np.sqrt(max(a_bar, 1e-8))
            x0 = np.clip(x0, -8.0, 8.0)
            if t == 0:
                x = x0
                break
            a_prev = float(self.alpha_bar[t - 1])
            frac = (1.0 - a_prev) / max(1.0 - a_bar, 1e-8) * (1.0 - a_bar / max(a_prev, 1e-8))
            sigma = eta * np.sqrt(max(frac, 0.0))
            direction = np.sqrt(max(1.0 - a_prev - sigma ** 2, 0.0)) * eps
            noise = rng.normal(size=x.shape) if eta > 0 else 0.0
            x = np.sqrt(a_prev) * x0 + direction + sigma * noise
        self.last_gate = gate.mean(axis=0)
        return np.clip(x, -8.0, 8.0), gate

    def mean(self, cond: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x, gate = self.ancestral(cond, rng, eta=0.0)
        self.last_gate = gate.mean(axis=0)
        return x


class ConditionalRealNVP:
    """Conditional real-NVP. The L5 architecture control, not the primary model."""

    def __init__(self, x_dim: int, cond_dim: int, hidden: int = 96, n_layers: int = 4, lr: float = 1e-3, seed: int = 2026):
        if x_dim % 2 != 0:
            raise ValueError("RealNVP wants an even state dimension")
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.layers = []
        for i in range(n_layers):
            which = "even" if i % 2 == 0 else "odd"
            self.layers.append(_Coupling(x_dim, cond_dim, hidden, seed + 17 * i, which, lr))
        self.loss_curve: list[float] = []
        self.last_gate = np.full(3, 1.0 / 3.0)

    def fit(self, x: np.ndarray, cond: np.ndarray, epochs: int, batch: int, seed: int) -> list[float]:
        rng = np.random.default_rng(seed)
        n = len(x)
        losses = []
        for _ in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            seen = 0
            for start in range(0, n, batch):
                idx = perm[start : start + batch]
                if len(idx) < 8:
                    continue
                loss = self._nll_and_step(x[idx], cond[idx])
                total += loss * len(idx)
                seen += len(idx)
            losses.append(total / max(seen, 1))
        self.loss_curve = losses
        return losses

    def _nll_and_step(self, x: np.ndarray, cond: np.ndarray) -> float:
        h = x
        logdet = np.zeros(len(x))
        for layer in self.layers:
            h, ld = layer.forward(h, cond)
            logdet = logdet + ld
        # L = mean(0.5 ||z||^2 - logdet)
        nll = 0.5 * np.sum(h * h, axis=1) - logdet
        n = len(x)
        g = h / n
        dlog = -np.ones(n) / n
        for layer in reversed(self.layers):
            g, _ = layer.backward(g, dlog)
        for layer in self.layers:
            layer.net.step()
        return float(np.mean(nll))

    def sample(self, cond: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z = rng.normal(size=(len(cond), self.x_dim))
        h = z
        for layer in reversed(self.layers):
            h = layer.inverse(h, cond)
        return np.clip(h, -8.0, 8.0)

    def mean(self, cond: np.ndarray, seed: int, n_avg: int = 4) -> np.ndarray:
        rng = np.random.default_rng(seed)
        acc = np.zeros((len(cond), self.x_dim))
        for _ in range(n_avg):
            acc += self.sample(cond, rng)
        return acc / n_avg


class _Coupling:
    def __init__(self, d: int, cond_dim: int, hidden: int, seed: int, which: str, lr: float):
        self.id_idx = np.arange(0, d, 2) if which == "even" else np.arange(1, d, 2)
        self.tr_idx = np.array([i for i in range(d) if i not in set(self.id_idx.tolist())])
        self.net = MLP(
            [len(self.id_idx) + cond_dim, hidden, hidden, 2 * len(self.tr_idx)],
            seed=seed,
            lr=lr,
        )
        self._cache = None

    def _split_raw(self, x_id: np.ndarray, cond: np.ndarray):
        raw = self.net.forward(np.concatenate([x_id, cond], axis=1))
        nb = len(self.tr_idx)
        s = np.tanh(raw[:, :nb])
        t = raw[:, nb:]
        return s, t

    def forward(self, x: np.ndarray, cond: np.ndarray):
        x_id = x[:, self.id_idx]
        x_tr = x[:, self.tr_idx]
        s, t = self._split_raw(x_id, cond)
        y = np.empty_like(x)
        y[:, self.id_idx] = x_id
        y[:, self.tr_idx] = x_tr * np.exp(np.clip(s, -8, 8)) + t
        self._cache = (x, cond, x_id, x_tr, s, t)
        return y, s.sum(axis=1)

    def inverse(self, y: np.ndarray, cond: np.ndarray) -> np.ndarray:
        y_id = y[:, self.id_idx]
        y_tr = y[:, self.tr_idx]
        s, t = self._split_raw(y_id, cond)
        x = np.empty_like(y)
        x[:, self.id_idx] = y_id
        x[:, self.tr_idx] = (y_tr - t) * np.exp(-np.clip(s, -8, 8))
        return x

    def backward(self, g_y: np.ndarray, dlogdet: np.ndarray):
        x, cond, x_id, x_tr, s, t = self._cache
        g_tr = g_y[:, self.tr_idx]
        g_id = g_y[:, self.id_idx]
        exp_s = np.exp(np.clip(s, -8, 8))
        d_t = g_tr
        d_s = g_tr * (x_tr * exp_s) + dlogdet[:, None]
        d_x_tr = g_tr * exp_s
        d_raw_s = d_s * (1.0 - s ** 2)
        d_raw = np.concatenate([d_raw_s, d_t], axis=1)
        d_h = self.net.backward(d_raw)
        d_x = np.zeros_like(x)
        d_x[:, self.id_idx] = g_id + d_h[:, : len(self.id_idx)]
        d_x[:, self.tr_idx] = d_x_tr
        d_cond = d_h[:, len(self.id_idx) :]
        return d_x, d_cond


class QuantileOutcome:
    """Histogram gradient boosting at the 0.1, 0.5 and 0.9 quantiles.

    Predictive draws use a normal scale matched to the central 80% interval.
    Quantile crossings are sorted before the scale is computed.
    """

    def __init__(self, max_iter: int = 60, seed: int = 2026):
        self.max_iter = max_iter
        self.seed = seed
        self.models: dict[float, HistGradientBoostingRegressor] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        for q in (0.1, 0.5, 0.9):
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=q,
                max_iter=self.max_iter,
                max_depth=4,
                learning_rate=0.08,
                early_stopping=False,
                random_state=self.seed,
            )
            model.fit(x, y)
            self.models[q] = model

    def median(self, x: np.ndarray) -> np.ndarray:
        """Conditional median. This is the point cross-section of the cohort."""
        return np.asarray(self.models[0.5].predict(x), dtype=np.float64)

    def sample(self, x: np.ndarray, rng: np.random.Generator, n_draws: int = 1) -> np.ndarray:
        lo = self.models[0.1].predict(x)
        mid = self.models[0.5].predict(x)
        hi = self.models[0.9].predict(x)
        stacked = np.sort(np.vstack([lo, mid, hi]), axis=0)
        lo, mid, hi = stacked[0], stacked[1], stacked[2]
        # N(0,1) central 80% width is 2 * 1.28155.
        scale = np.clip((hi - lo) / (2.0 * 1.28155156554), 0.05, None)
        noise = rng.normal(size=(n_draws, len(x)))
        return mid[None, :] + scale[None, :] * noise
