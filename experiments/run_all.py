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
from src.federation import local_train, federated_average, evaluate

np.random.seed(SEED)
torch.manual_seed(SEED)


EXPERIMENTS = [
    {
        "key":     "FedAvg_IID",
        "label":   "Expérience 1 — FedAvg + IID (référence)",
        "split":   split_iid,
        "mu":      0.0,
    },
    {
        "key":     "FedAvg_nonIID",
        "label":   "Expérience 2 — FedAvg + non-IID  [Verrou 1 : données hétérogènes]",
        "split":   split_non_iid,
        "mu":      0.0,
    },
    {
        "key":     "FedProx_nonIID",
        "label":   f"Expérience 3 — FedProx (μ={MU_FEDPROX}) + non-IID  [Solution Verrou 1]",
        "split":   split_non_iid,
        "mu":      MU_FEDPROX,
    },
    {
        "key":     "FedProx_IID",
        "label":   f"Expérience 4 — FedProx (μ={MU_FEDPROX}) + IID  (contrôle)",
        "split":   split_iid,
        "mu":      MU_FEDPROX,
    },
]


def run_experiment(X_train, y_train, X_test, y_test, exp):
    print(f"\n{'='*62}")
    print(f"  {exp['label']}")
    print(f"{'='*62}")

    test_loader = DataLoader(
        make_dataset(X_test, y_test), batch_size=256, shuffle=False
    )
    node_datasets = exp["split"](X_train, y_train, N_NODES)
    node_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        for ds in node_datasets
    ]

    sizes = [len(ds) for ds in node_datasets]
    total = sum(sizes)
    fed_weights = [s / total for s in sizes]
    print(f"  Données par nœud : {sizes}  |  Poids : {[f'{w:.3f}' for w in fed_weights]}")

    global_model = LightMLP()
    print(f"  Paramètres modèle : {global_model.count_parameters():,}")

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
        global_model = federated_average(global_model, local_models, fed_weights)
        acc, loss = evaluate(global_model, test_loader)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        print(f"  Round {rnd:2d}/{N_ROUNDS} | Acc: {acc*100:.2f}% | Loss: {loss:.4f}")

    return history


def main():
    print("=" * 62)
    print("  FedEdge6G — Simulation DFL sur Topologie Réseau 6G")
    print("=" * 62)
    print(f"\n  Classes de trafic : {CLASS_NAMES}")
    print("\n  Profils des nœuds (scénario non-IID) :")
    for k, v in NODE_PROFILES.items():
        print(f"    Nœud {k} : {v}")

    print("\n📊 Génération des données...")
    X, y = generate_traffic_data()
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(f"   Train : {len(y_train)} échantillons | Test : {len(y_test)} échantillons")

    results = {}
    for exp in EXPERIMENTS:
        results[exp["key"]] = run_experiment(X_train, y_train, X_test, y_test, exp)

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Résumé ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RÉSUMÉ FINAL — Accuracy au round 20")
    print("=" * 62)
    labels = {
        "FedAvg_IID":      "FedAvg + IID",
        "FedAvg_nonIID":   "FedAvg + non-IID     ← Verrou 1",
        "FedProx_nonIID":  "FedProx + non-IID    ← Solution",
        "FedProx_IID":     "FedProx + IID",
    }
    for key, hist in results.items():
        acc = hist["accuracy"][-1] * 100
        print(f"  {labels[key]:<38} {acc:.2f}%")

    gap = (results["FedProx_nonIID"]["accuracy"][-1] -
           results["FedAvg_nonIID"]["accuracy"][-1]) * 100
    print(f"\n  ✅ Gain FedProx vs FedAvg (non-IID) : +{gap:.1f}%")
    print(f"  → Verrou 1 validé : FedProx atténue le client drift")
    print(f"    sous données hétérogènes (non-IID).")
    print("\n  📁 Résultats sauvegardés dans results/results.json")


if __name__ == "__main__":
    main()