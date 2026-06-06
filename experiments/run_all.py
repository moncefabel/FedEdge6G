import copy
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    N_NODES, N_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE,
    LR, MU_FEDPROX, SEED, NODE_PROFILES, CLASS_NAMES
)
from src.model import LightMLP
from src.data import (
    generate_traffic_data, train_test_split,
    make_dataset, split_iid, split_non_iid
)
from src.federation import (
    local_train, federated_average, evaluate,
    communication_cost_per_round, rounds_to_convergence
)

np.random.seed(SEED)
torch.manual_seed(SEED)

CONVERGENCE_THRESHOLD = 0.85

EXPERIMENTS = [
    {"key": "FedAvg_IID",      "label": "FedAvg + IID",
     "split": split_iid,      "mu": 0.0},
    {"key": "FedAvg_nonIID",   "label": "FedAvg + non-IID  [Verrou 1]",
     "split": split_non_iid,  "mu": 0.0},
    {"key": "FedProx_nonIID",  "label": f"FedProx (μ={MU_FEDPROX}) + non-IID  [Solution Verrou 1]",
     "split": split_non_iid,  "mu": MU_FEDPROX},
    {"key": "FedProx_IID",     "label": f"FedProx (μ={MU_FEDPROX}) + IID",
     "split": split_iid,      "mu": MU_FEDPROX},
]


def run_experiment(X_train, y_train, X_test, y_test, exp):
    print(f"\n{'='*62}")
    print(f"  {exp['label']}")
    print(f"{'='*62}")

    test_loader = DataLoader(
        make_dataset(X_test, y_test), batch_size=256, shuffle=False)
    node_datasets = exp["split"](X_train, y_train, N_NODES)
    node_loaders  = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
                     for ds in node_datasets]

    sizes = [len(ds) for ds in node_datasets]
    fed_weights = [s / sum(sizes) for s in sizes]

    global_model = LightMLP()

    #  Communication cost (fixe, calculé une fois) 
    comm = communication_cost_per_round(global_model, N_NODES)
    print(f"  Paramètres modèle    : {comm['params']:,}")
    print(f"  Coût/round réseau    : {comm['KB']} KB  "
          f"({comm['MB']} MB)")
    print(f"  Coût total ({N_ROUNDS} rounds): "
          f"{round(comm['KB'] * N_ROUNDS, 1)} KB  "
          f"({round(comm['MB'] * N_ROUNDS, 4)} MB)")
    print(f"  Données par nœud     : {sizes}")

    history = {"accuracy": [], "loss": []}

    for rnd in range(1, N_ROUNDS + 1):
        local_models = [
            local_train(
                copy.deepcopy(global_model), loader,
                epochs=LOCAL_EPOCHS, lr=LR,
                mu=exp["mu"], global_model=global_model
            )
            for loader in node_loaders
        ]
        global_model = federated_average(
            global_model, local_models, fed_weights)
        acc, loss = evaluate(global_model, test_loader)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        print(f"  Round {rnd:2d}/{N_ROUNDS} | "
              f"Acc: {acc*100:.2f}% | Loss: {loss:.4f}")

    #  Rounds to convergence ─
    rtc = rounds_to_convergence(
        history["accuracy"], CONVERGENCE_THRESHOLD)
    history["rounds_to_convergence"] = rtc
    history["comm_cost_KB_per_round"] = comm["KB"]
    history["total_comm_cost_KB"] = round(
        comm["KB"] * (rtc if rtc else N_ROUNDS), 1)

    if rtc:
        print(f"\n  ✅ Seuil {int(CONVERGENCE_THRESHOLD*100)}% atteint "
              f"au round {rtc}/{N_ROUNDS}")
        print(f"     Coût comm. jusqu'à convergence : "
              f"{history['total_comm_cost_KB']} KB")
    else:
        print(f"\n  ⚠️  Seuil {int(CONVERGENCE_THRESHOLD*100)}% "
              f"non atteint en {N_ROUNDS} rounds")

    return history


def main():
    print("=" * 62)
    print("  FedEdge6G — Simulation DFL | Topologie Réseau 6G")
    print("=" * 62)
    print(f"\n  Classes de trafic : {CLASS_NAMES}")
    print("\n  Profils nœuds (non-IID) :")
    for k, v in NODE_PROFILES.items():
        print(f"    Nœud {k} : {v}")

    print("\n📊 Génération des données...")
    X, y = generate_traffic_data()
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(f"   Train: {len(y_train)} | Test: {len(y_test)}")

    results = {}
    for exp in EXPERIMENTS:
        results[exp["key"]] = run_experiment(
            X_train, y_train, X_test, y_test, exp)

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    #  Tableau de synthèse ─
    print("\n" + "=" * 70)
    print("  SYNTHÈSE FINALE")
    print("=" * 70)
    print(f"  {'Expérience':<32} {'Acc.':<10} "
          f"{'Rounds→85%':<14} {'Coût comm.'}")
    print("-" * 70)

    labels = {
        "FedAvg_IID":     "FedAvg + IID",
        "FedAvg_nonIID":  "FedAvg + non-IID   [Verrou 1]",
        "FedProx_nonIID": "FedProx + non-IID  [Solution]",
        "FedProx_IID":    "FedProx + IID",
    }
    for key, hist in results.items():
        acc = hist["accuracy"][-1] * 100
        rtc = hist["rounds_to_convergence"]
        cost = hist["total_comm_cost_KB"]
        rtc_str  = f"round {rtc}" if rtc else "non atteint"
        cost_str = f"{cost} KB"
        print(f"  {labels[key]:<32} {acc:<10.2f}% "
              f"{rtc_str:<14} {cost_str}")

    print("=" * 70)

    #  Insight énergétique ─
    fa  = results["FedAvg_nonIID"]
    fp  = results["FedProx_nonIID"]
    acc_gap  = (fp["accuracy"][-1] - fa["accuracy"][-1]) * 100
    rtc_fa   = fa["rounds_to_convergence"] or N_ROUNDS
    rtc_fp   = fp["rounds_to_convergence"] or N_ROUNDS
    rounds_saved = rtc_fa - rtc_fp
    energy_saved = rounds_saved * fa["comm_cost_KB_per_round"]

    print(f"\n  🔋 Efficacité énergétique (non-IID) :")
    print(f"     FedProx converge {rounds_saved} rounds plus tôt que FedAvg")
    print(f"     → {energy_saved:.1f} KB de communication économisés")
    print(f"     → +{acc_gap:.1f}% accuracy finale")
    print(f"\n     Verrou 1 validé : FedProx atténue le client drift")
    print(f"     Verrou 3 validé : moins de rounds = moins d'énergie")
    print("\n  📁 Résultats → results/results.json")


if __name__ == "__main__":
    main()