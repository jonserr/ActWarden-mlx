"""Risk scorer: compact MLP trained with mlx.nn, exported for the C++ runtime.

The export contract (see docs/architecture/overview.md):
  input  : float32 [batch, feature_dim] in spec order
  output : float32 [batch, num_heads] sigmoid probabilities in risk_heads order
Weights are baked into the .mlxfn as constants; the runtime never loads
separate weight files. Retraining => re-export.

Requires the `train` extra (pip install -e actwarden/python[train]).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

DEFAULT_HIDDEN = 64


class RiskScorer(nn.Module):
    """16 -> 64 -> 64 -> 4 sigmoid heads. Small on purpose: this sits in the
    tool-call hot path with a sub-millisecond CPU budget."""

    def __init__(self, feature_dim: int, num_heads: int, hidden: int = DEFAULT_HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(feature_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_heads)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
        return mx.sigmoid(self.out(x))


def bce_loss(model: RiskScorer, x: mx.array, y: mx.array) -> mx.array:
    p = mx.clip(model(x), 1e-6, 1.0 - 1e-6)
    return -mx.mean(y * mx.log(p) + (1.0 - y) * mx.log(1.0 - p))


def train(
    model: RiskScorer,
    x_train: mx.array,
    y_train: mx.array,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Minimal training loop. Data generation/labeling is the Day-1 slice task
    (see overview.md); this loop is intentionally boring."""
    mx.random.seed(seed)
    opt = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, bce_loss)
    n = x_train.shape[0]
    history: list[float] = []
    for _ in range(epochs):
        perm = mx.array(mx.random.permutation(n))
        epoch_loss = 0.0
        batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss, grads = loss_and_grad(model, x_train[idx], y_train[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            epoch_loss += loss.item()
            batches += 1
        history.append(epoch_loss / max(batches, 1))
    return history


def export_for_runtime(model: RiskScorer, path: str | Path, feature_dim: int) -> None:
    """Export the scoring graph for mx::import_function in C++.

    Parameters are captured as constants inside the exported graph (same
    pattern as examples/export/eval_mlp.py upstream).
    """
    mx.eval(model.parameters())

    def forward(x: mx.array) -> mx.array:
        return model(x)

    example = mx.zeros((1, feature_dim), dtype=mx.float32)
    mx.export_function(str(path), forward, example)

    # Round-trip sanity check before handing the artifact to the runtime.
    imported = mx.import_function(str(path))
    (got,) = imported(example)
    if not mx.allclose(got, model(example)).item():
        raise RuntimeError(f"export round-trip mismatch for {path}")
