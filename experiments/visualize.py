import json, os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLASS_NAMES, NODE_PROFILES, N_ROUNDS

COLORS = {
    "FedAvg_IID":     "#2196F3",
    "FedAvg_nonIID":  "#F44336",
    "FedProx_nonIID": "#4CAF50",
    "FedProx_IID":    "#FF9800",
}
DISPLAY = {
    "FedAvg_IID":     "FedAvg + IID",
    "FedAvg_nonIID":  "FedAvg + non-IID",
    "FedProx_nonIID": "FedProx + non-IID",
    "FedProx_IID":    "FedProx + IID",
}
LS = {
    "FedAvg_IID": "-", "FedAvg_nonIID": "--",
    "FedProx_nonIID": "-", "FedProx_IID": "--",
}


def plot(results_path="results/results.json",
         out="results/dfl_results.png"):
    with open(results_path) as f:
        results = json.load(f)

    rounds = list(range(1, N_ROUNDS + 1))
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.35)

    #  1. Accuracy 
    ax1 = fig.add_subplot(gs[0, :2])
    for key, hist in results.items():
        ax1.plot(rounds, [a * 100 for a in hist["accuracy"]],
                 color=COLORS[key], linestyle=LS[key],
                 linewidth=2.3, marker="o", markersize=4,
                 label=DISPLAY[key])

    # Ligne seuil convergence 85%
    ax1.axhline(85, color="#888", linestyle=":", linewidth=1.2,
                label="Seuil convergence (85%)")

    # Annotations rounds to convergence
    for key in ["FedAvg_nonIID", "FedProx_nonIID"]:
        rtc = results[key].get("rounds_to_convergence")
        if rtc:
            acc_at_rtc = results[key]["accuracy"][rtc - 1] * 100
            ax1.axvline(rtc, color=COLORS[key],
                        linestyle=":", linewidth=1, alpha=0.6)
            ax1.annotate(f"R{rtc}",
                xy=(rtc, acc_at_rtc),
                xytext=(rtc + 0.3, acc_at_rtc - 8),
                fontsize=8, color=COLORS[key],
                arrowprops=dict(arrowstyle="->",
                                color=COLORS[key], lw=1))

    ax1.set_xlabel("Round de communication", fontsize=10)
    ax1.set_ylabel("Accuracy globale (%)", fontsize=10)
    ax1.set_title("Convergence — Accuracy par round\n"
                  "(lignes pointillées verticales = round où 85% est atteint)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 108)

    # ── 2. Profils nœuds 
    ax2 = fig.add_subplot(gs[0, 2])
    node_dist = [
        [0.35, 0.35, 0.05, 0.05, 0.05, 0.15],
        [0.05, 0.05, 0.35, 0.05, 0.35, 0.15],
        [0.05, 0.05, 0.05, 0.35, 0.05, 0.45],
        [1/6]*6,
    ]
    node_labels = [f"Nœud {i}" for i in range(4)]
    palette = ["#E91E63","#9C27B0","#3F51B5",
               "#00BCD4","#4CAF50","#FF9800"]
    bottom = np.zeros(4)
    for c, (cls, col) in enumerate(zip(CLASS_NAMES, palette)):
        vals = [d[c] for d in node_dist]
        ax2.bar(node_labels, vals, bottom=bottom,
                color=col, label=cls,
                edgecolor="white", linewidth=0.7)
        bottom += np.array(vals)
    ax2.set_ylabel("Proportion du trafic", fontsize=9)
    ax2.set_title("Distribution non-IID\npar nœud edge",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(axis="y", alpha=0.2)

    #  3. Loss 
    ax3 = fig.add_subplot(gs[1, 0])
    for key, hist in results.items():
        ax3.plot(rounds, hist["loss"],
                 color=COLORS[key], linestyle=LS[key],
                 linewidth=2.2, marker="s", markersize=3.5,
                 label=DISPLAY[key])
    ax3.set_xlabel("Round", fontsize=10)
    ax3.set_ylabel("Loss (Cross-Entropy)", fontsize=10)
    ax3.set_title("Convergence — Loss", fontsize=10,
                  fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    #  4. Rounds to convergence (barplot) 
    ax4 = fig.add_subplot(gs[1, 1])
    keys = list(results.keys())
    rtcs = []
    for k in keys:
        rtc = results[k].get("rounds_to_convergence")
        rtcs.append(rtc if rtc else N_ROUNDS)

    bars = ax4.bar([DISPLAY[k] for k in keys], rtcs,
                   color=[COLORS[k] for k in keys],
                   edgecolor="white", linewidth=1.5)
    for bar, val, k in zip(bars, rtcs, keys):
        rtc_real = results[k].get("rounds_to_convergence")
        label = f"R{val}" if rtc_real else f">{N_ROUNDS}"
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 label, ha="center", va="bottom", fontsize=9)

    ax4.set_ylabel("Rounds jusqu'à 85%", fontsize=10)
    ax4.set_title("Rounds to Convergence (85%)\n"
                  "← Moins = plus efficace énergétiquement",
                  fontsize=10, fontweight="bold")
    ax4.tick_params(axis="x", labelsize=7.5)
    ax4.grid(axis="y", alpha=0.2)

    #  5. Communication cost total 
    ax5 = fig.add_subplot(gs[1, 2])
    costs = [results[k]["total_comm_cost_KB"] for k in keys]
    bars5 = ax5.bar([DISPLAY[k] for k in keys], costs,
                    color=[COLORS[k] for k in keys],
                    edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars5, costs):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{val:.0f} KB",
                 ha="center", va="bottom", fontsize=8.5)
    ax5.set_ylabel("KB échangés jusqu'à convergence", fontsize=10)
    ax5.set_title("Coût de communication total\n"
                  "(upload + download, jusqu'au seuil 85%)",
                  fontsize=10, fontweight="bold")
    ax5.tick_params(axis="x", labelsize=7.5)
    ax5.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "FedEdge6G — Simulation DFL sur Topologie Réseau 6G à 4 Nœuds\n"
        "FedAvg vs FedProx | IID vs non-IID | "
        "Classification trafic 6G synthétique\n"
        "Thèse CIFRE Orange Innovation — Projet TREES (réf. 2026-51929)",
        fontsize=11, fontweight="bold", y=1.01
    )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Figure sauvegardée : {out}")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists("results/results.json"):
        print("Lance d'abord : python -m experiments.run_all")
    else:
        plot()