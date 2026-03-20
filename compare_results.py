import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD FILES (4 experiments)
# -----------------------------
files = {
    "Baseline (No Quant)": "baseline_NQ_results.csv",
    "Baseline (Quant)": "baseline_Q_results.csv",
    "Proposed (No Quant)": "proposed_NQ_results.csv",
    "Proposed (Quant)": "proposed_Q_results.csv"
}

data = {}

for key, file in files.items():
    df = pd.read_csv(file)
    data[key] = df

# Ensure same length
min_len = min(len(df) for df in data.values())
for key in data:
    data[key] = data[key].iloc[:min_len]

# -----------------------------
# 1. ACCURACY COMPARISON (RAW)
# -----------------------------
plt.figure()
for key, df in data.items():
    plt.plot(df["accuracy"], label=key)

plt.title("Accuracy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("comparison_accuracy.png")
plt.show()

# -----------------------------
# 2. TOTAL ENERGY (RAW)
# -----------------------------
plt.figure()
for key, df in data.items():
    plt.plot(df["total_energy"], label=key)

plt.title("Total Energy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.savefig("comparison_total_energy.png")
plt.show()

# -----------------------------
# 3. COMPUTATION ENERGY (RAW)
# -----------------------------
plt.figure()
for key, df in data.items():
    plt.plot(df["compute_energy"], label=key)

plt.title("Computation Energy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.savefig("comparison_compute_energy.png")
plt.show()

# -----------------------------
# 4. COMMUNICATION ENERGY (RAW)
# -----------------------------
plt.figure()
for key, df in data.items():
    plt.plot(df["communication_energy"], label=key)

plt.title("Communication Energy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.savefig("comparison_communication_energy.png")
plt.show()

# -----------------------------
# 5. FINAL SUMMARY (UNCHANGED)
# -----------------------------
def last_avg(series):
    return series.tail(10).mean()

print("\n===== FINAL COMPARISON =====")

for key, df in data.items():

    print(f"\n--- {key} ---")

    print("Accuracy:", last_avg(df["accuracy"]))
    print("Computation Energy:", last_avg(df["compute_energy"]))
    print("Communication Energy:", last_avg(df["communication_energy"]))
    print("Total Energy:", last_avg(df["total_energy"]))