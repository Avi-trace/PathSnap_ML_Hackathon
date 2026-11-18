"""Small PyTorch MLP with a scikit-learn-style fit / predict_proba interface and an epoch history."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TorchMLP:
    def __init__(self, hidden=(128, 64), epochs=15, lr=1e-3, weight_decay=1e-5, batch=4096, val_frac=0.05, seed=0):
        self.hidden, self.epochs, self.lr, self.wd = tuple(hidden), epochs, lr, weight_decay
        self.batch, self.val_frac, self.seed = batch, val_frac, seed
        self.history_ = {"train_loss": [], "val_loss": []}
        self.net_ = None

    def _build(self, n_in: int) -> nn.Sequential:
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        return nn.Sequential(*layers, nn.Linear(d, 1))

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xt = torch.from_numpy(((X - self.mu_) / self.sd_).astype(np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        val = rng.random(len(Xt)) < self.val_frac
        tr_idx, va_idx = torch.from_numpy(np.flatnonzero(~val)), torch.from_numpy(np.flatnonzero(val))
        self.net_ = self._build(X.shape[1])
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            self.net_.train()
            perm = tr_idx[torch.randperm(len(tr_idx))]
            tot = 0.0
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                loss = lossf(self.net_(Xt[b]).squeeze(1), yt[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            self.net_.eval()
            with torch.no_grad():
                vl = lossf(self.net_(Xt[va_idx]).squeeze(1), yt[va_idx]).item() if len(va_idx) else float("nan")
            self.history_["train_loss"].append(tot / len(perm))
            self.history_["val_loss"].append(vl)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = torch.from_numpy(((X - self.mu_) / self.sd_).astype(np.float32))
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(torch.cat([self.net_(Xt[i:i + 262144]).squeeze(1) for i in range(0, len(Xt), 262144)])).numpy()
        return np.column_stack([1 - p, p])
