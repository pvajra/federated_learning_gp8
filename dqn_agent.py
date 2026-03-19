import torch
import torch.nn as nn
import torch.optim as optim

class DQNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.fc(x)

class DQNAgent:
    def __init__(self):
        self.model = DQNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    def select_client_score(self, state):
        with torch.no_grad():
            return self.model(torch.FloatTensor(state).unsqueeze(0)).item()
