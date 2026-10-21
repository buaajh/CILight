"""CILight terminal-value components extracted without algorithm changes."""
import numpy as np
import torch
from torch import nn


GAMMA = .98


COST_SCALE = 100.


def terminal_features(topo, queue, arrivals, phase, remaining):
    q = np.maximum(np.asarray(queue), 0.)
    a = np.maximum(np.asarray(arrivals), 0.)
    cap = np.maximum(topo.capacity, 1.)
    occ = q / cap
    serve = topo.service[:, int(phase)]
    return np.asarray([occ.mean(), occ.max(), (occ**2).mean(), q.sum()/100,
        a.sum()/100, cap.mean()/100, (q*serve).sum()/100,
        (q*(1-serve)).sum()/100, (a*serve).sum()/100, serve.mean(),
        len(q)/32, max(remaining, 0.)/3600], dtype=np.float32)


class TerminalCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 32), nn.Tanh(),
                                 nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        # True finite-episode terminal states have zero continuation cost.
        return self.net(x.clamp(-10, 10)).squeeze(-1) * (x[..., -1] > 0)

    def numpy_weights(self):
        return [(m.weight.detach().cpu().numpy().astype(float).copy(),
                 m.bias.detach().cpu().numpy().astype(float).copy())
                for m in self.net if isinstance(m, nn.Linear)]


class NumpyTerminalValue:
    """Pickle-safe, immutable rollout snapshot for Stage-5 worker payloads."""
    def __init__(self, weights, beta, remaining, horizon):
        self.weights, self.beta = weights, float(beta)
        self.remaining, self.horizon = remaining, horizon

    def __call__(self, topo, queue, arrivals, phase):
        if self.remaining <= 0 or self.beta == 0:
            return 0.
        x = np.clip(terminal_features(topo, queue, arrivals, phase,
                                     self.remaining).astype(float), -10, 10)
        for i, (w, b) in enumerate(self.weights):
            x = w @ x + b
            if i < len(self.weights)-1:
                x = np.tanh(x)
        return self.beta * GAMMA**self.horizon * COST_SCALE * float(np.clip(x[0], 0, 100))

