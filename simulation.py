import flwr as fl
import torch
import matplotlib.pyplot as plt
import random
import pandas as pd
import os

from client import FlowerClient
from dataset import load_datasets
from model import Net


NUM_CLIENTS = 50
NUM_ROUNDS = 30
CLIENTS_PER_ROUND = 10

def run_experiment(mode, use_quantization):
    client_stats = {}
    compute_energy_log = []
    communication_energy_log = []
    total_energy_log = []
    accuracy_history = []

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

            if mode == "baseline":
                return super().configure_fit(server_round, parameters, client_manager)

            clients = list(client_manager.all().values())
            scored = []

            for c in clients:
                cid = c.cid

                if cid not in client_stats:
                    score = random.random()
                else:
                    score = compute_score(client_stats[cid])

                scored.append((c, score))

            scored.sort(key=lambda x: x[1], reverse=True)

            # ------------------ DIVERSITY ------------------
            top_k = scored[:30]
            selected = random.sample([c for c, _ in top_k], CLIENTS_PER_ROUND)

            return [(c, fl.common.FitIns(parameters, {})) for c in selected]

        def aggregate_fit(self, server_round, results, failures):

            agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

            comp, comm, total = 0, 0, 0

            for client, res in results:
                client_stats[client.cid] = res.metrics

                comp += res.metrics.get("compute_energy", 0)
                comm += res.metrics.get("communication_energy", 0)
                total += res.metrics.get("total_energy", 0)

            compute_energy_log.append(comp)
            communication_energy_log.append(comm)
            total_energy_log.append(total)

            print(f"\nRound {server_round}")
            print("Compute:", comp)
            print("Comm:", comm)
            print("Total:", total)

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
    min_len = min(len(accuracy_history), len(total_energy_log))

    df = pd.DataFrame({
        "accuracy": accuracy_history[:min_len],
        "compute_energy": compute_energy_log[:min_len],
        "communication_energy": communication_energy_log[:min_len],
        "total_energy": total_energy_log[:min_len]
    })

    run_name = f"{mode}_{'Q' if use_quantization else 'NQ'}"
    df.to_csv(f"{run_name}_results.csv", index=False)
    print(f"{run_name}_results.csv saved")

    return  {
        "name": run_name,
        "accuracy": accuracy_history[:min_len],
        "total_energy": total_energy_log[:min_len]
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


# ------------------ EVALUATION ------------------
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


# ------------------ SCORE ------------------
def compute_score(stats):

    return (
        0.2 * stats["battery"]
        + 0.2 * stats["cpu"]
        + 0.2 * (1 - stats["dropout"])
        - 0.2 * stats["compute_energy"]
        - 0.2 * stats["communication_energy"]
    )


# # ------------------ PLOT ------------------
# plt.figure()
# plt.plot(accuracy_history)
# plt.title("Accuracy")
# plt.savefig(f"{MODE}_accuracy.png")
#
# plt.figure()
# plt.plot(total_energy_log)
# plt.title("Energy")
# plt.savefig(f"{MODE}_energy.png")


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