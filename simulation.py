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

        def aggregate_fit(self, server_round, results, failures):
            agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
            comp, comm, total = 0, 0, 0

            for client, res in results:
                client_stats[client.cid] = res.metrics
                comp += res.metrics.get("compute_energy", 0)
                comm += res.metrics.get("communication_energy", 0)
                total += res.metrics.get("total_energy", 0)

            # Carbon Calculation
            curr_c = round_carbon.get(server_round, 300.0)
            carbon_emitted = total * curr_c

            compute_energy_log.append(comp)
            communication_energy_log.append(comm)
            total_energy_log.append(total)
            carbon_emissions_log.append(carbon_emitted)

            print(f"\nRound {server_round}")
            print("Total Energy:", total)
            print("Carbon Emitted:", carbon_emitted)

            return agg_params, agg_metrics

    # ------------------ RUN ------------------
    strategy = EnergyStrategy(
        fraction_fit=0.2,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    # ------------------ SAVE ------------------
    min_len = min(
        len(accuracy_history), len(total_energy_log), len(carbon_emissions_log),
        len(compute_energy_log), len(communication_energy_log)
    )
    run_name = f"{mode}_{'Q' if use_quantization else 'NQ'}"
    
    # FIX: Keep original compute/comm columns so the leader's compare_results.py doesn't crash
    df = pd.DataFrame({
        "accuracy": accuracy_history[:min_len],
        "compute_energy": compute_energy_log[:min_len],
        "communication_energy": communication_energy_log[:min_len],
        "total_energy": total_energy_log[:min_len],
        "carbon_emissions": carbon_emissions_log[:min_len]
    })
    df.to_csv(f"{run_name}_results.csv", index=False)
    print(f"{run_name}_results.csv saved")

    return  {
        "name": run_name,
        "accuracy": accuracy_history[:min_len],
        "total_energy": total_energy_log[:min_len],
        "carbon_emissions": carbon_emissions_log[:min_len]
    }

trainloaders, testloader = load_datasets(NUM_CLIENTS)

# ------------------ DEVICE PROFILES ------------------
device_types = ["sensor", "mobile", "edge"]
client_profiles = {}
for cid in range(NUM_CLIENTS):
    device = random.choice(device_types)
    if device == "sensor":
        profile = {"battery": 0.4, "cpu_factor": 0.5, "compression": 0.4, "dropout": 0.3}
    elif device == "mobile":
        profile = {"battery": 0.7, "cpu_factor": 0.8, "compression": 0.7, "dropout": 0.15}
    else:
        profile = {"battery": 1.0, "cpu_factor": 1.2, "compression": 1.0, "dropout": 0.05}
    client_profiles[cid] = profile

def test_model(parameters):
    model = Net()
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

# ------------------ RUN ------------------
experiments = [
    ("baseline", False),
    ("baseline", True),
    ("proposed", False),
    ("proposed", True)
]

all_results = {}
for mode, use_quantization in experiments:
    key = f"{mode}_{'Q' if use_quantization else 'NQ'}"
    all_results[key] = run_experiment(mode, use_quantization)


# ------------------ PLOT ------------------
def moving_avg(values, window=5):
    return pd.Series(values).rolling(window, min_periods=1).mean()

plt.figure()
for key, result in all_results.items():
    plt.plot(moving_avg(result["accuracy"]), label=key)
plt.title("Accuracy Comparison (5-Round Avg)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_comp.png")

plt.figure()
for key, result in all_results.items():
    plt.plot(moving_avg(result["total_energy"]), label=key)
plt.title("Total Energy Comparison (5-Round Avg)")
plt.xlabel("Round")
plt.ylabel("Total Energy")
plt.legend()
plt.savefig("energy_comp.png")
