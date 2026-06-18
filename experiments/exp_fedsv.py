import sys
import os
sys.path.insert(0, '/home/claude/FedEdge6G')

import copy
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import (
    N_NODES, N_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR, MU_FEDPROX,
    N_SAMPLES, TEST_RATIO, N_FEATURES, N_CLASSES, SEED, NODE_PROFILES, CLASS_NAMES
)
from src.model import LightMLP
from src.data import generate_traffic_data, train_test_split, split_non_iid, make_dataset
from src.federation import local_train, federated_average, evaluate, communication_cost_per_round, rounds_to_convergence
from src.federation_sv import federated_average_sv, compute_loo_sv


def run_experiment(name, use_fedprox=False, use_sv=False, mu=MU_FEDPROX, n_rounds=N_ROUNDS, seed=SEED):
    """
    Lance un round complet de fédération.

    Args:
        name       : nom de l'expérience
        use_fedprox: activer le terme proximal FedProx
        use_sv     : activer l'agrégation pondérée FedSV
        mu         : coefficient proximal (FedProx)
    """
    # --- Données ---
    X, y = generate_traffic_data(seed=seed)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=seed)
    node_datasets = split_non_iid(X_train, y_train, N_NODES, seed=seed)
    test_dataset = make_dataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256)

    node_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        for ds in node_datasets
    ]

    #  Modèle global initial 
    global_model = LightMLP()
    torch.manual_seed(seed)

    history = {
        "accuracy":   [],
        "loss":       [],
        "sv_scores":  [],  # par round [liste de 4 SV]
        "sv_weights": [],
    }

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for r in range(n_rounds):
        # 1. Entraînement local
        local_models = []
        for i in range(N_NODES):
            m = copy.deepcopy(global_model)
            if use_fedprox:
                m = local_train(m, node_loaders[i], LOCAL_EPOCHS, LR,
                                mu=mu, global_model=global_model)
            else:
                m = local_train(m, node_loaders[i], LOCAL_EPOCHS, LR)
            local_models.append(m)

        # 2. Agrégation
        if use_sv:
            global_model, sv_scores, sv_weights = federated_average_sv(
                global_model, local_models, test_loader
            )
            history["sv_scores"].append(sv_scores)
            history["sv_weights"].append(sv_weights)
        else:
            global_model = federated_average(global_model, local_models)
            history["sv_scores"].append([None]*N_NODES)
            history["sv_weights"].append([0.25]*N_NODES)

        # 3. Évaluation
        acc, loss = evaluate(global_model, test_loader)
        history["accuracy"].append(acc)
        history["loss"].append(loss)

        if use_sv and history["sv_scores"][-1][0] is not None:
            sv = history["sv_scores"][-1]
            sv_str = " | ".join([f"N{i}:{sv[i]:+.4f}" for i in range(N_NODES)])
            print(f"Round {r+1:2d} | Acc: {acc:.4f} | SV: [{sv_str}]")
        else:
            print(f"Round {r+1:2d} | Acc: {acc:.4f}")

    # 4. Métriques finales
    comm_cost = communication_cost_per_round(global_model, N_NODES)
    r_conv = rounds_to_convergence(history["accuracy"], threshold=0.85)

    print(f"\n--- Résultats finaux : {name} ---")
    print(f"Accuracy finale    : {history['accuracy'][-1]*100:.2f}%")
    print(f"Rounds → 85%       : {r_conv}")
    print(f"Comm. cost / round : {comm_cost['KB']:.1f} KB")

    return history, comm_cost, r_conv


def analyze_sv_scores(sv_history):
    """
    Analyse les SV scores moyens par nœud sur tous les rounds.
    Montre quels nœuds contribuent positivement / négativement.
    """
    n_rounds = len(sv_history)
    n_nodes  = len(sv_history[0])
    means = []
    for i in range(n_nodes):
        vals = [sv_history[r][i] for r in range(n_rounds) if sv_history[r][i] is not None]
        means.append(np.mean(vals) if vals else 0.0)
    return means



if __name__ == "__main__":

    print("\n" + "="*60)
    print("  EXPÉRIENCE COMPARATIVE : FedAvg vs FedProx vs FedSV")
    print("  Scénario : non-IID (4 nœuds 6G hétérogènes)")
    print("="*60)

    # Exp A : FedAvg baseline (non-IID)
    hist_avg, cost_avg, conv_avg = run_experiment(
        "FedAvg + non-IID (baseline)", use_fedprox=False, use_sv=False
    )

    # Exp B : FedProx (non-IID, μ=0.1)
    hist_prox, cost_prox, conv_prox = run_experiment(
        f"FedProx + non-IID (μ={MU_FEDPROX})", use_fedprox=True, use_sv=False
    )

    # Exp C : FedSV (non-IID) — agrégation pondérée Shapley
    hist_sv, cost_sv, conv_sv = run_experiment(
        "FedSV + non-IID (LOO Shapley)", use_fedprox=False, use_sv=True
    )

    # Exp D : FedSV + FedProx (combinaison)
    hist_sv_prox, cost_sv_prox, conv_sv_prox = run_experiment(
        f"FedSV + FedProx + non-IID (μ={MU_FEDPROX})", use_fedprox=True, use_sv=True
    )


    print("\n" + "="*60)
    print("  TABLEAU COMPARATIF")
    print("="*60)
    print(f"{'Méthode':<35} {'Acc finale':>10} {'→85%':>6} {'KB/round':>10}")
    print("-"*65)

    results = [
        ("FedAvg + non-IID",             hist_avg,      conv_avg,      cost_avg),
        (f"FedProx + non-IID (μ=0.1)",   hist_prox,     conv_prox,     cost_prox),
        ("FedSV + non-IID",              hist_sv,       conv_sv,       cost_sv),
        ("FedSV + FedProx + non-IID",    hist_sv_prox,  conv_sv_prox,  cost_sv_prox),
    ]

    for name, hist, conv, cost in results:
        acc = hist["accuracy"][-1]
        print(f"{name:<35} {acc*100:>9.2f}% {str(conv):>6} {cost['KB']:>8.1f} KB")

    #  ANALYSE SV SCORES 
    print("\n" + "="*60)
    print("  ANALYSE DES SHAPLEY VALUES PAR NŒUD (FedSV)")
    print("="*60)
    print("  SV_i > 0 : nœud contribue positivement au modèle global")
    print("  SV_i < 0 : nœud dégrade le modèle → exclu de l'agrégation")
    print("-"*60)

    sv_means_sv      = analyze_sv_scores(hist_sv["sv_scores"])
    sv_means_sv_prox = analyze_sv_scores(hist_sv_prox["sv_scores"])

    print("\nSV moyen (FedSV seul) :")
    for i, (sv_mean, profile) in enumerate(zip(sv_means_sv, NODE_PROFILES.values())):
        flag = "✅" if sv_mean >= 0 else "⚠️ "
        print(f"  Nœud {i} ({profile[:30]}) : SV = {sv_mean:+.5f} {flag}")

    print("\nSV moyen (FedSV + FedProx) :")
    for i, (sv_mean, profile) in enumerate(zip(sv_means_sv_prox, NODE_PROFILES.values())):
        flag = "✅" if sv_mean >= 0 else "⚠️ "
        print(f"  Nœud {i} ({profile[:30]}) : SV = {sv_mean:+.5f} {flag}")

    