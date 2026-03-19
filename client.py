import flwr as fl
import torch
import random
import os
from model import Net

ALPHA = 0.000001
BETA = 0.0000005

# FIX: Safely fallback to CPU to avoid OOM crashes during heavy parallel simulation
_device_str = os.getenv("FLOWER_CLIENT_DEVICE", "cpu").lower()
if _device_str == "cuda" and not torch.cuda.is_available():
    _device_str = "cpu"
DEVICE = torch.device(_device_str)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, device_profile, use_quantization=False):
        self.model = Net().to(DEVICE) # Ensure device compatibility
        self.trainloader = trainloader
        self.profile = device_profile
        self.use_quantization = use_quantization

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device=DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        dropout_prob = self.profile.get("dropout", 0.0)
        cpu_factor = self.profile.get("cpu_factor", 1.0)
        compression = self.profile.get("compression", 1.0)

        # ------------------ DROPOUT ------------------
        if random.random() < dropout_prob:
            params = self.get_parameters({})
            if self.use_quantization:
                params = self.quantize_parameters(params)
            return params, 0, {
                "compute_energy": 0.0, 
                "communication_energy": 0.0, 
                "total_energy": 0.0,
                "battery": self.profile.get("battery", 1.0), 
                "cpu": cpu_factor,
                "compression": compression, 
                "data": 0, 
                "dropout": dropout_prob,
                "quantized": int(self.use_quantization)
            }

        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.train()

        training_steps = 0
        max_steps = int(len(self.trainloader) * cpu_factor)

        for i, (data, target) in enumerate(self.trainloader):
            if i >= max_steps: break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            training_steps += 1

        # ------------------ ENERGY CALCULATION ------------------
        cpu_cycles = training_steps * cpu_factor * 1000
        compute_energy = cpu_cycles * ALPHA

        model_size = sum(p.numel() for p in self.model.parameters())
        width = 16 if self.use_quantization else 32
        quantization_factor = width / 32.0

        transmitted_params = model_size * compression * quantization_factor
        communication_energy = transmitted_params * BETA
        total_energy = compute_energy + communication_energy

        metrics = {
            "compute_energy": compute_energy,
            "communication_energy": communication_energy,
            "total_energy": total_energy,
            "battery": self.profile.get("battery", 1.0),
            "cpu": cpu_factor,
            "data": len(self.trainloader.dataset),
            "dropout": dropout_prob,
            "quantized": int(self.use_quantization)
        }

        params = self.get_parameters({})
        if self.use_quantization:
            params = self.quantize_parameters(params)

        return params, len(self.trainloader.dataset), metrics

    def quantize_parameters(self, parameters):
        return [p.astype("float16") for p in parameters]

    def evaluate(self, parameters, config):
        """Standard Flower evaluate method to prevent runtime errors."""
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader.dataset), {}
