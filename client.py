import flwr as fl
import torch
import random
from model import Net

ALPHA, BETA = 0.000001, 0.0000005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, profile):
        self.model = Net().to(DEVICE)
        self.trainloader = trainloader
        self.profile = profile

    def get_parameters(self, config): return [val.cpu().numpy() for val in self.model.state_dict().values()]
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if random.random() < self.profile["dropout"]:
            metrics = {
                "total_energy": 0.0,
                "compute_energy": 0.0,
                "communication_energy": 0.0,
                "battery": self.profile.get("battery", 1.0),
                "cpu": self.profile.get("cpu_factor", 1.0),
                "dropout": 1.0,
            }
            return self.get_parameters({}), 0, metrics
        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        steps = 0
        max_steps = int(len(self.trainloader) * self.profile["cpu_factor"])
        for i, (data, target) in enumerate(self.trainloader):
            if i >= max_steps: break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad(); loss = torch.nn.CrossEntropyLoss()(self.model(data), target)
            loss.backward(); optimizer.step(); steps += 1
        
        comp_e = (steps * self.profile["cpu_factor"] * 1000) * ALPHA
        comm_e = (sum(p.numel() for p in self.model.parameters()) * self.profile["compression"]) * BETA
        metrics = {
            "total_energy": comp_e + comm_e,
            "compute_energy": comp_e,
            "communication_energy": comm_e,
            "battery": self.profile.get("battery", 1.0),
            "cpu": self.profile.get("cpu_factor", 1.0),
            "dropout": 0.0,
        }
        return self.get_parameters({}), len(self.trainloader.dataset), metrics
