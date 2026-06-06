import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLASS_NAMES, NODE_PROFILES

COLORS = {
    "FedAvg_IID":     "#2196F3",
    "FedAvg_nonIID":  "#F44336",
    "FedProx_nonIID": "#4CAF50",
    "FedProx_IID":    "#FF9800",
}
DISPLAY_LABELS = {
    "FedAvg_IID":     "FedAvg + IID",
    "FedAvg_nonIID":  "FedAvg + non-IID",
    "FedProx_nonIID": "FedProx + non-IID",
    "FedProx_IID":    "FedProx + IID",
}
LINESTYLES = {
    "FedAvg_IID":     "-",
    "FedAvg_nonIID":  "--",
    "FedProx_nonIID": "-",
    "FedProx_IID":    "--",
}


def plot(results_path="results/results.json", out="results/dfl_results.png"):
    with open(results_path) as f:
        results = json.load(f)

    n_rounds = len(next(iter(results.values()))["accuracy"])
    rounds = list(range(1, n_rounds + 1))

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Graphique 1 : Accuracy ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for key, hist in results.items():
        ax1.plot(rounds, [a * 100 for a in hist["accuracy"]],
                 color=COLORS[key], linestyle=LINESTYLES[key],
                 linewidth=2.2, marker="o", markersize=3.5,
                 label=DISPLAY_LABELS[key])

    # Annotation gap FedProx vs FedAvg en non-IID
    fa = results["FedAvg_nonIID"]["accuracy"][-1] * 100
    fp = results["FedProx_nonIID"]["accuracy"][-1] * 100
    mid = (fa + fp) / 2
    ax1.annotate("", xy=(n_rounds, fp), xytext=(n_rounds, fa),
                 arrowprops=dict(arrowstyle="<->", color="#888", lw=1.5))
    ax1.text(n_rounds - 1.5, mid, f"+{fp-fa:.1f}%",
             color="#333", fontsize=8.5, ha="right", va="center",
             bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.2"))

    ax1.set_xlabel("Round de communication", fontsize=10)
    ax1.set_ylabel("Accuracy globale (%)", fontsize=10)
    ax1.set_title("Convergence — Accuracy", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(0, 105)

    # ── Graphique 2 : Loss ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for key, hist in results.items():
        ax2.plot(rounds, hist["loss"],
                 color=COLORS[key], linestyle=LINESTYLES[key],
                 linewidth=2.2, marker="s", markersize=3.5,
                 label=DISPLAY_LABELS[key])
    ax2.set_xlabel("Round de communication", fontsize=10)
    ax2.set_ylabel("Loss (Cross-Entropy)", fontsize=10)
    ax2.set_title("Convergence — Loss", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.25)

    # ── Graphique 3 : Barplot comparatif ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    keys = list(results.keys())
    accs = [results[k]["accuracy"][-1] * 100 for k in keys]
    colors = [COLORS[k] for k in keys]
    bars = ax3.bar([DISPLAY_LABELS[k] for k in keys], accs,
                   color=colors, edgecolor="white", linewidth=1.5)
    for bar, acc in zip(bars, accs):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    ax3.set_ylim(0, 110)
    ax3.set_ylabel("Accuracy finale (%)", fontsize=10)
    ax3.set_title("Accuracy finale — Round 20", fontsize=11, fontweight="bold")
    ax3.tick_params(axis="x", labelsize=8)
    ax3.grid(axis="y", alpha=0.25)

    # ── Graphique 4 : Profils des nœuds (non-IID) ──────────
    ax4 = fig.add_subplot(gs[1, 1])
    node_distributions = [
        [0.35, 0.35, 0.05, 0.05, 0.05, 0.15],   # Résidentiel
        [0.05, 0.05, 0.35, 0.05, 0.35, 0.15],   # Industriel
        [0.05, 0.05, 0.05, 0.35, 0.05, 0.45],   # Dense urbain
        [1/6]*6,                                   # Mixte
    ]
    node_labels = [f"Nœud {i}" for i in range(4)]
    bottom = np.zeros(4)
    palette = ["#E91E63", "#9C27B0", "#3F51B5", "#00BCD4", "#4CAF50", "#FF9800"]
    for c, (cls_name, color) in enumerate(zip(CLASS_NAMES, palette)):
        vals = [d[c] for d in node_distributions]
        ax4.bar(node_labels, vals, bottom=bottom, color=color,
                label=cls_name, edgecolor="white", linewidth=0.8)
        bottom += np.array(vals)
    ax4.set_ylabel("Proportion du trafic", fontsize=10)
    ax4.set_title("Profils des nœuds — Distribution non-IID", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=7.5, loc="upper right")
    ax4.grid(axis="y", alpha=0.25)

    # ── Titre global ─────────────────────────────────────────
    fig.suptitle(
        "FedEdge6G — Simulation DFL sur Topologie Réseau 4 Nœuds\n"
        "FedAvg vs FedProx | Distribution IID vs non-IID | Trafic 6G Synthétique\n"
        "Thèse CIFRE Orange Innovation — Projet TREES (réf. 2026-51929)",
        fontsize=11, fontweight="bold", y=1.01
    )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Figure sauvegardée : {out}")
    plt.close()


if __name__ == "__main__":
    # Lance les expériences si pas de résultats
    if not os.path.exists("results/results.json"):
        print("Pas de résultats trouvés. Lance d'abord : python -m experiments.run_all")
    else:
        plot()