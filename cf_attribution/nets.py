"""Small MLP with Adam. Used by the conditional DDPM and the RealNVP."""

from __future__ import annotations

import numpy as np


class MLP:
    def __init__(
        self,
        sizes: list[int],
        seed: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for din, dout in zip(self.sizes[:-1], self.sizes[1:]):
            # He init. Last layer is linear and smaller.
            scale = np.sqrt(2.0 / din) if dout != self.sizes[-1] else np.sqrt(1.0 / din)
            self.W.append(rng.normal(0.0, scale, size=(din, dout)).astype(np.float64))
            self.b.append(np.zeros(dout, dtype=np.float64))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self._acts: list[np.ndarray] = []
        self._pre: list[np.ndarray] = []
        self._dW: list[np.ndarray] = []
        self._db: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        self._acts = [h]
        self._pre = []
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w + b
            self._pre.append(z)
            h = z if i == len(self.W) - 1 else np.maximum(z, 0.0)
            self._acts.append(h)
        return h

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate a scalar loss.

        ``grad`` is dL/dY for the scalar being optimized, with the same
        shape as the forward output. The returned tensor is dL/dX.
        There is no extra 1/n inside this method: put it in ``grad`` if
        the scalar is a batch mean.
        """
        g = np.asarray(grad, dtype=np.float64)
        dW: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        db: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                g = g * (self._pre[i] > 0.0)
            dW[i] = self._acts[i].T @ g
            db[i] = g.sum(axis=0)
            g = g @ self.W[i].T
        self._dW = dW
        self._db = db
        return g

    def step(self, clip: float = 5.0) -> None:
        if not self._dW:
            return
        # Global grad clip across layers.
        sq = 0.0
        for gw, gb in zip(self._dW, self._db):
            sq += float(np.sum(gw * gw) + np.sum(gb * gb))
        scale = 1.0
        if sq > clip * clip:
            scale = clip / np.sqrt(sq)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            gw = self._dW[i] * scale + self.weight_decay * self.W[i]
            gb = self._db[i] * scale
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * gw
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * (gw * gw)
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * (gb * gb)
            mhat = self.mW[i] / (1 - b1**self.t)
            vhat = self.vW[i] / (1 - b2**self.t)
            bhat = self.mb[i] / (1 - b1**self.t)
            vbhat = self.vb[i] / (1 - b2**self.t)
            self.W[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.b[i] -= self.lr * bhat / (np.sqrt(vbhat) + eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def time_embedding(t: np.ndarray, dim: int = 16) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / max(half, 1))
    ang = t[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
