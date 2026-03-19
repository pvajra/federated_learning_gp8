import flwr as fl
import torch
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

from client import FlowerClient
from dataset import load_datasets
from model import Net
# --- NEW IMPORTS FOR DRL ---
from carbon_logic import CarbonGridSimulator
from dqn_agent import DQNAgent

NUM_CLIENTS = 50
NUM_ROUNDS = 30 # Matching leader's new length
CLIENTS_PER_ROUND = 10

def run_experiment(mode, use_quantization):
    # FIX: Initialize DRL components INSIDE the run so they reset for each experiment
    carbon_sim = CarbonGridSimulator()
    drl_agent = DQNAgent()

    client_stats = {}
    compute_energy_log = []
    communication_energy_log = []
    total_energy_log = []
    carbon_emissions_log = []
    accuracy_history = []
    round_carbon = {}

    def evaluate(server_round, parameters, config):
        acc = test_model(parameters)
        accuracy_history.append(acc)
        print(f"Round {server_round} Accuracy: {acc}")
        return 0.0, {"accuracy": acc}

    def client_fn(cid: str):
        return FlowerClient(trainloaders[int(cid)], client_profiles[int(cid)], use_quantization=use_quantization)

    # ------------------ STRATEGY ------------------
    class EnergyStrategy(fl.server.strategy.FedAvg):

        def configure_fit(self, server_round, parameters, client_manager):
            # FIX: Calculate round carbon BEFORE the baseline return so baseline has accurate logging
            round_carbon[server_round] = np.mean([carbon_sim.get_carbon_intensity(i, server_round) for i in range(3)])

            if mode == "baseline":
                return super().configure_fit(server_round, parameters, client_manager)

            # FIX: Sort clients by CID so zone assignment is deterministic
            clients = sorted(list(client_manager.all().values()), key=lambda c: int(c.cid))
            scored = []

            # --- DRL CARBON-AWARE SELECTION ---
            for idx, c in enumerate(clients):
                zone_id = idx % 3
                client_carbon = carbon_sim.get_carbon_intensity(zone_id, server_round)
                
                stats = client_stats.get(c.cid, {"battery": 0.8, "cpu": 0.1})
                
                state = [
                    client_carbon / 600.0, 
                    stats.get("battery", 0.8), 
                    stats.get("cpu", 0.1), 
                    server_round / NUM_ROUNDS
                ]
                score = drl_agent.select_client_score(state)
                scored.append((c, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            
            selected = [x[0] for x in scored[:CLIENTS_PER_ROUND]]
            return [(c, fl.common.FitIns(parameters, {})) for c in selected]
