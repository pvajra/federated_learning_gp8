# Contribution Notes — Animesh Kumar

## What This Folder Contains

This folder holds my individual contribution to the Group 8 FL project,
implemented and verified independently via Google Colab Pro (T4 GPU).

### Files

| File | Description |
|------|-------------|
| `FL_Group8_Colab_FINAL.ipynb` | Complete self-contained Colab notebook — runs all 4 experiments end to end |
| `results/baseline_NQ_results.csv` | Baseline, float32 — 30 rounds |
| `results/baseline_Q_results.csv` | Baseline + quantisation — 30 rounds |
| `results/proposed_NQ_results.csv` | Proposed DQN carbon-aware, float32 — 30 rounds |
| `results/proposed_Q_results.csv` | Proposed DQN carbon-aware + quantisation — 30 rounds |
| `results/comparison_summary.csv` | Aggregated metrics across all 4 runs |
| `results/experiment_results.png` | Accuracy / Energy / Carbon line plots (3-round moving avg) |
| `results/energy_breakdown.png` | Compute vs Communication energy breakdown |
| `results/main_comparison.png` | 4-panel comparison across all metrics |
| `results/summary_bars.png` | Bar chart summary — final accuracy, total energy, total carbon |

---

## My Contribution — Carbon-Aware DQN Client Selection

Building on the group's baseline random selection strategy, I designed and
implemented a **Deep Q-Network (DQN) based carbon-aware client selection**
strategy embedded directly into the simulation pipeline.

### Components implemented

**1. Stochastic Carbon Grid Simulator (`carbon_logic`)**
- Three geographic energy zones: green (Nordic, ~80 gCO₂eq/kWh),
  mixed (UK/DE, ~250), coal (PL/IN, ~550)
- Sinusoidal daily pattern + Gaussian noise:
  `intensity(t) = base + amplitude × sin(2πt/24) + N(0, σ²)`
- Per-round caching for consistency; reset between experiment runs

**2. DQN Agent (`DQNAgent`)**
- State vector (dim=5): `[norm_carbon, battery, cpu_norm, dropout_risk, round_progress]`
- Online training via experience replay (buffer=4,000 transitions)
- Target network synced every 5 steps; gradient clipping (max_norm=1.0)
- Epsilon-greedy exploration: ε starts at 1.0 and decays by 0.905 per round (≈0.05 after 30 rounds; lower-bounded at 0.05)
- Reward: `+1.0 × Δaccuracy − 0.4 × norm_energy − 0.6 × norm_carbon`

**3. Energy-Aware Strategy (inside `FL_Group8_Colab_FINAL.ipynb` Cell 8)**
- Overrides `FedAvg.configure_fit` for DQN-based selection
- Implements proposal Eq. 1 participation score (W1=0.30, W2=0.30, W3=0.20, W4=0.20)
- Implements proposal Eq. 2–4 joint energy cost model
- Tracks `compute_energy`, `communication_energy`, `total_energy`, `carbon_emissions` per round
- UUID-safe CID mapping (compatible with Flower ≥1.7)

---

## Key Results (30 rounds, MNIST, 50 clients, 10/round)

| Run | Final Accuracy | Peak Accuracy | Total Energy | Carbon (gCO₂eq) |
|-----|---------------|--------------|-------------|-----------------|
| baseline_NQ | 98.23% | 98.33% | 223.5 J | 0.01852 |
| baseline_Q  | 98.34% | 98.34% | 135.9 J | 0.01150 |
| proposed_NQ | 98.23% | 98.42% | 216.8 J | **0.01574** |
| proposed_Q  | **98.47%** | **98.47%** | **135.0 J** | 0.01195 |

### Headline findings

- **15.0% carbon reduction** — proposed_NQ vs baseline_NQ at identical accuracy (98.23%)
- **~40% communication energy reduction** — from float16 quantisation
- **Best overall accuracy 98.47%** — proposed_Q (DQN + quantisation combined)
- **Faster convergence** — proposed_NQ reaches 97% at round 11 vs baseline round 14

---

## How to Run

1. Open `FL_Group8_Colab_FINAL.ipynb` in Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells top to bottom (Cell 1 through 11)
4. In Cell 9, set `QUICK_MODE = True` for a 2-run quick test (~10 min)
   or `QUICK_MODE = False` for all 4 experiments (~20 min)

All outputs are saved to `/content/FL_outputs/` and downloaded as a zip in Cell 11.

---

## CSV Compatibility

All result CSVs use the exact same column format as the group's `compare_results.py`:
```
round,accuracy,compute_energy,communication_energy,total_energy,carbon_emissions
```

No changes to any existing group files are required.

---

## Relation to Proposal

This contribution directly addresses:
- **RQ1** — joint optimisation of computation and communication energy
- **RQ2** — energy-aware client selection influence on convergence and accuracy
- **Gap addressed** — adds carbon as an explicit third optimisation dimension
  beyond the proposal's computation + communication scope

*Animesh Kumar — Newcastle University*
