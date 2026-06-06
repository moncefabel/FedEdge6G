import json, os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLASS_NAMES, NODE_PROFILES, N_ROUNDS

COLORS = {
    "FedAvg_IID":             "#2196F3",
    "FedAvg_nonIID":          "#F44336",
    "FedProx_nonIID":         "#4CAF50",
    "FedProx_nonIID_partial": "#9C27B0",
}
DISPLAY = {
    "FedAvg_IID":             "FedAvg + IID",
    "FedAvg_nonIID":          "FedAvg + non-IID",
    "FedProx_nonIID":         "FedProx + non-IID",
    "FedProx_nonIID_partial": "FedProx + non-IID + Partial (3/4)",
}
LS = {
    "FedAvg_IID": "-", "FedAvg_nonIID": "--",
    "FedProx_nonIID": "-", "FedProx_nonIID_partial": "-.",
}


def plot(results_path="results/results.json",
         out="results/dfl_results.png"):
    with open(results_path) as f:
        results = json.load(f)

    mu_data   = results.pop("mu_sensitivity", {})
    main_keys = list(results.keys())
    rounds    = list(range(1, N_ROUNDS + 1))

    fig = plt.figure(figsize=(20, 13))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. Accuracy convergence ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for key in main_keys:
        ax1.plot(rounds, [a * 100 for a in results[key]["accuracy"]],
                 color=COLORS[key], linestyle=LS[key],
                 linewidth=2.3, marker="o", markersize=3.5,
                 label=DISPLAY[key])
    ax1.axhline(85, color="#888", linestyle=":", linewidth=1.2,
                label="Seuil 85%")
    for key in ["FedAvg_nonIID", "FedProx_nonIID", "FedProx_nonIID_partial"]:
        rtc = results[key].get("rounds_to_convergence")
        if rtc:
            ax1.axvline(rtc, color=COLORS[key], linestyle=":", alpha=0.5, lw=1)
            ax1.text(rtc + 0.1,
                     results[key]["accuracy"][rtc-1]*100 - 6,
                     f"R{rtc}", fontsize=7.5, color=COLORS[key])
    ax1.set_xlabel("Round", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_title("Convergence — Accuracy\n"
                  "Exp 1→4 : FedAvg/FedProx × IID/non-IID × participation complète/partielle",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(0, 108)

    # ── 2. Profils nœuds ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    node_dist = [
        [0.35,0.35,0.05,0.05,0.05,0.15],
        [0.05,0.05,0.35,0.05,0.35,0.15],
        [0.05,0.05,0.05,0.35,0.05,0.45],
        [1/6]*6,
    ]
    palette = ["#E91E63","#9C27B0","#3F51B5","#00BCD4","#4CAF50","#FF9800"]
    bottom  = np.zeros(4)
    for c, (cls, col) in enumerate(zip(CLASS_NAMES, palette)):
        vals = [d[c] for d in node_dist]
        ax2.bar([f"N{i}" for i in range(4)], vals, bottom=bottom,
                color=col, label=cls, edgecolor="white", linewidth=0.7)
        bottom += np.array(vals)
    ax2.set_title("Distribution non-IID\npar nœud edge 6G",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(axis="y", alpha=0.2)

    # ── 3. Loss ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for key in main_keys:
        ax3.plot(rounds, results[key]["loss"],
                 color=COLORS[key], linestyle=LS[key],
                 linewidth=2.2, marker="s", markersize=3.5,
                 label=DISPLAY[key])
    ax3.set_xlabel("Round", fontsize=10)
    ax3.set_ylabel("Loss (Cross-Entropy)", fontsize=10)
    ax3.set_title("Convergence — Loss", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.2)

    # ── 4. Pareto front : Accuracy vs Comm Cost ───────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for key in main_keys:
        h    = results[key]
        acc  = h["accuracy"][-1] * 100
        cost = h.get("total_comm_cost_KB", 0)
        ax4.scatter(cost, acc, color=COLORS[key], s=180,
                    zorder=5, edgecolors="white", linewidth=1.5)
        ax4.annotate(DISPLAY[key],
                     xy=(cost, acc),
                     xytext=(cost + 5, acc - 0.8),
                     fontsize=7.5, color=COLORS[key])

    # Ligne Pareto idéale
    ax4.annotate("", xy=(450, 99), xytext=(100, 99),
                 arrowprops=dict(arrowstyle="-", color="#ddd", lw=1))
    ax4.text(100, 99.2, "Pareto idéal →\n(haute acc., faible coût)",
             fontsize=7, color="#999")

    ax4.set_xlabel("Coût communication jusqu'à convergence (KB)", fontsize=10)
    ax4.set_ylabel("Accuracy finale (%)", fontsize=10)
    ax4.set_title("Pareto Front\nAccuracy vs Coût de communication\n"
                  "[Verrou 3 : arbitrage multi-objectif]",
                  fontsize=10, fontweight="bold")
    ax4.grid(True, alpha=0.2)
    ax4.set_ylim(88, 100)

    # ── 5. Rounds to convergence ──────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    rtcs   = [results[k].get("rounds_to_convergence") or N_ROUNDS
               for k in main_keys]
    bars   = ax5.bar([DISPLAY[k] for k in main_keys], rtcs,
                     color=[COLORS[k] for k in main_keys],
                     edgecolor="white", linewidth=1.5)
    for bar, val, k in zip(bars, rtcs, main_keys):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f"R{val}", ha="center", va="bottom", fontsize=9)
    ax5.set_ylabel("Rounds jusqu'à 85%", fontsize=10)
    ax5.set_title("Rounds to Convergence\n← moins = plus économe",
                  fontsize=10, fontweight="bold")
    ax5.tick_params(axis="x", labelsize=7)
    ax5.grid(axis="y", alpha=0.2)

    # ── 6. Sensibilité μ ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    if mu_data:
        mu_vals  = [float(k) for k in mu_data.keys()]
        accs     = [v["final_accuracy"] * 100 for v in mu_data.values()]
        rtcs_mu  = [v.get("rounds_to_convergence") or N_ROUNDS
                    for v in mu_data.values()]

        ax6_twin = ax6.twinx()
        line1, = ax6.plot(mu_vals, accs, "o-", color="#2196F3",
                          linewidth=2.5, markersize=8, label="Accuracy finale (%)")
        line2, = ax6_twin.plot(mu_vals, rtcs_mu, "s--", color="#FF5722",
                               linewidth=2, markersize=7,
                               label="Rounds to convergence")

        for x, y in zip(mu_vals, accs):
            ax6.annotate(f"{y:.1f}%", (x, y),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8, color="#2196F3")
        for x, y in zip(mu_vals, rtcs_mu):
            ax6_twin.annotate(f"R{y}", (x, y),
                              textcoords="offset points", xytext=(0, -14),
                              ha="center", fontsize=8, color="#FF5722")

        ax6.set_xlabel("Valeur de μ (coefficient proximal FedProx)", fontsize=11)
        ax6.set_ylabel("Accuracy finale (%)", fontsize=10, color="#2196F3")
        ax6_twin.set_ylabel("Rounds to convergence", fontsize=10, color="#FF5722")
        ax6.set_title(
            "Analyse de sensibilité μ — Impact du terme proximal FedProx (non-IID)\n"
            "μ=0 : FedAvg standard | μ petit : légère contrainte | "
            "μ grand : contrainte forte → convergence ralentie",
            fontsize=10, fontweight="bold")
        ax6.legend(handles=[line1, line2], fontsize=9, loc="lower left")
        ax6.grid(True, alpha=0.2)

        # Annotation zone optimale
        opt_mu = mu_vals[np.argmax(accs)]
        ax6.axvspan(0.0, 0.15, alpha=0.08, color="green")
        ax6.text(0.07, min(accs) + 0.5, "Zone optimale\nμ ∈ [0, 0.1]",
                 fontsize=8, color="green", ha="center")

    fig.suptitle(
        "FedEdge6G — Simulation DFL sur Topologie Réseau 6G à 4 Nœuds\n"
        "FedAvg vs FedProx | IID vs non-IID | Participation partielle | "
        "Pareto front | Sensibilité μ\n"
        "Thèse CIFRE Orange Innovation — Projet TREES (réf. 2026-51929)",
        fontsize=11, fontweight="bold", y=1.01
    )

    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Dashboard sauvegardé : {out}")
    plt.close()


if __name__ == "__main__":
    if not os.path.exists("results/results.json"):
        print("Lance d'abord : python -m experiments.run_all")
    else:
        plot()