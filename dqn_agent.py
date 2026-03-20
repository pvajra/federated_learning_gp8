# =============================================================================
# dqn_agent.py
# Deep Q-Network for carbon-aware federated client selection.
# Animesh Kumar — Newcastle University
#
# State vector (dim=5):
#   [norm_carbon, battery, cpu_norm, dropout_risk, round_progress]
#
# Trained online via experience replay after every FL aggregation round.
# Reward: +1.0 x delta_accuracy - 0.4 x norm_energy - 0.6 x norm_carbon
# =============================================================================
import random
from collections import deque
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

_S = 5; _H = 128; _LR = 3e-4; _G = 0.95
_EPS_START = 1.0; _EPS_MIN = 0.05; _EPS_DECAY = 0.905
_BS = 64; _CAP = 4000; _TSYNC = 5

class _QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_S, _H), nn.LayerNorm(_H), nn.ReLU(),
            nn.Linear(_H, _H), nn.LayerNorm(_H), nn.ReLU(),
            nn.Linear(_H, 1),
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    """
    Online DQN for FL client selection.
    Per round: select_clients → store → learn → decay_epsilon.
    State: [norm_carbon, battery, cpu_norm, dropout_risk, round_progress]
    """
    def __init__(self, seed=42):
        torch.manual_seed(seed); random.seed(seed)
        self.q    = _QNet(); self.tgt = _QNet()
        self.tgt.load_state_dict(self.q.state_dict()); self.tgt.eval()
        self.opt  = optim.Adam(self.q.parameters(), lr=_LR)
        self.mem  = deque(maxlen=_CAP)
        self.eps  = _EPS_START; self._step = 0

    def _score(self, s):
        t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): return float(self.q(t).item())

    def select_clients(self, states: Dict[str, List[float]], k: int) -> List[str]:
        cids = list(states.keys()); k = min(k, len(cids))
        if random.random() < self.eps:
            return random.sample(cids, k)
        return sorted(cids, key=lambda c: self._score(states[c]), reverse=True)[:k]

    def store(self, s, r, ns, done):
        self.mem.append((s, r, ns, float(done)))

    def learn(self):
        if len(self.mem) < _BS: return
        b = random.sample(self.mem, _BS)
        ss, rr, ns, dd = zip(*b)
        s = torch.tensor(np.array(ss), dtype=torch.float32)
        r = torch.tensor(rr, dtype=torch.float32).unsqueeze(1)
        n = torch.tensor(np.array(ns), dtype=torch.float32)
        d = torch.tensor(dd, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad(): tq = r + _G * self.tgt(n) * (1 - d)
        loss = F.mse_loss(self.q(s), tq)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()
        self._step += 1
        if self._step % _TSYNC == 0: self.tgt.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        self.eps = max(_EPS_MIN, self.eps * _EPS_DECAY)

    @staticmethod
    def compute_reward(norm_e, norm_c, acc_gain):
        return 1.0 * acc_gain - 0.4 * norm_e - 0.6 * norm_c

if __name__ == "__main__":
    _a = DQNAgent(0)
    print(f'✅ DQNAgent | test Q: {_a._score([0.3,0.7,0.8,0.1,0.5]):.4f} | ε={_a.eps}')
    del _a
