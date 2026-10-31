"""Create presentation-ready plots from the experiment CSV outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "zkdp_fl_results.csv"
CLIENT_FILE = RESULTS_DIR / "client_results_trust_seed_42.csv"


def main():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"{RESULTS_FILE} not found. Run code/zkdp_fl_experiment.py first."
        )

    df = pd.read_csv(RESULTS_FILE)

    # Average convergence curves across the three experiment seeds.
    mean_curves = (
        df.groupby(["mode", "round"])[["loss", "accuracy"]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    for mode, label in [("fixed", "Fixed noise"), ("trust", "Trust-aware noise")]:
        subset = mean_curves[mean_curves["mode"] == mode]
        plt.plot(subset["round"], subset["loss"], marker="o", label=label)
    plt.xlabel("Federated round")
    plt.ylabel("Global test loss")
    plt.title("Global Loss Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "loss_convergence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for mode, label in [("fixed", "Fixed noise"), ("trust", "Trust-aware noise")]:
        subset = mean_curves[mean_curves["mode"] == mode]
        plt.plot(subset["round"], subset["accuracy"] * 100, marker="o", label=label)
    plt.xlabel("Federated round")
    plt.ylabel("Global test accuracy (%)")
    plt.title("Global Accuracy Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "accuracy_convergence.png", dpi=300)
    plt.close()

    # Final mean and standard deviation across seeds.
    final_round = int(df["round"].max())
    final_rows = df[df["round"] == final_round]
    summary = final_rows.groupby("mode")["accuracy"].agg(["mean", "std"])

    plt.figure(figsize=(7, 5))
    order = [m for m in ["fixed", "trust"] if m in summary.index]
    labels = ["Fixed noise" if m == "fixed" else "Trust-aware noise" for m in order]
    values = [summary.loc[m, "mean"] * 100 for m in order]
    errors = [summary.loc[m, "std"] * 100 for m in order]
    plt.bar(labels, values, yerr=errors, capsize=5)
    plt.ylabel("Final global accuracy (%)")
    plt.title("Final Accuracy Comparison (mean ± SD)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_accuracy_comparison.png", dpi=300)
    plt.close()

    # Trust-to-noise mapping used in the experiment.
    trust = pd.DataFrame({
        "client": ["C0", "C1", "C2", "C3", "C4"],
        "trust": [0.2, 0.4, 0.6, 0.8, 1.0],
        "sigma": [0.017, 0.014, 0.011, 0.008, 0.005],
    })
    plt.figure(figsize=(7, 5))
    plt.plot(trust["trust"], trust["sigma"], marker="o")
    for _, row in trust.iterrows():
        plt.annotate(row["client"], (row["trust"], row["sigma"]), xytext=(5, 6), textcoords="offset points")
    plt.xlabel("Client trust score")
    plt.ylabel("Gaussian noise standard deviation (sigma)")
    plt.title("Trust Score–to–Noise Mapping")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "trust_noise_mapping.png", dpi=300)
    plt.close()

    # Client-level results are saved by the experiment for trust mode, seed 42.
    if CLIENT_FILE.exists():
        clients = pd.read_csv(CLIENT_FILE)
        plt.figure(figsize=(7, 5))
        plt.bar(clients["client"].astype(str), clients["accuracy"] * 100)
        plt.xlabel("Client")
        plt.ylabel("Client test accuracy (%)")
        plt.title("Client-Level Accuracy (Trust-Aware, Seed 42)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "client_level_accuracy.png", dpi=300)
        plt.close()
    else:
        print("Client-level CSV not found; skipping client-level plot.")

    print(f"Plots saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
