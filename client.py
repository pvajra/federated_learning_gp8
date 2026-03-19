import flwr as fl
import torch
import random
import numpy as np
from model import Net

ALPHA, BETA = 0.000001, 0.0000005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, profile, use_quantization=False):
        self.model = Net().to(DEVICE)
        self.trainloader = trainloader
        self.profile = profile
        self.use_quantization = use_quantization

    def get_parameters(self, config): 
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        # Fix: Ensure tensors are created on the correct DEVICE to avoid mismatch
        state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def quantize_parameters(self, parameters):
        return [p.astype("float16") for p in parameters]

    def fit(self, parameters, config):
        # 1. DROPOUT LOGIC
        if random.random() < self.profile["dropout"]:
            metrics = {
                "total_energy": 0.0,
                "compute_energy": 0.0,
                "communication_energy": 0.0,
                "battery": self.profile.get("battery", 1.0),
                "cpu": self.profile.get("cpu_factor", 1.0),
                "dropout": 1.0,
                "quantized": int(self.use_quantization)
            }
            params = self.get_parameters({})
            if self.use_quantization:
                params = self.quantize_parameters(params)
            return params, 0, metrics

        # 2. TRAINING SETUP
        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        steps = 0
        max_steps = int(len(self.trainloader) * self.profile["cpu_factor"])
        
        for i, (data, target) in enumerate(self.trainloader):
            if i >= max_steps: break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(self.model(data), target)
            loss.backward()
            optimizer.step()
            steps += 1
        
        # 3. ENERGY CALCULATION (Carbon-Aware Logic)
        comp_e = (steps * self.profile["cpu_factor"] * 1000) * ALPHA
        
        # Adjust communication energy based on quantization
        model_size = sum(p.numel() for p in self.model.parameters())
        q_factor = 0.5 if self.use_quantization else 1.0
        comm_e = (model_size * self.profile["compression"] * q_factor) * BETA
        
        metrics = {
            "total_energy": comp_e + comm_e,
            "compute_energy": comp_e,
            "communication_energy": comm_e,
            "battery": self.profile.get("battery", 1.0),
            "cpu": self.profile.get("cpu_factor", 1.0),
            "dropout": 0.0,
            "quantized": int(self.use_quantization)
        }
        
        params = self.get_parameters({})
        if self.use_quantization:
            params = self.quantize_parameters(params)
            
        return params, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader.dataset), {}