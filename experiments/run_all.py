import copy, json, sys, os
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    communication_cost_per_round, rounds_to_convergence,
    select_participants, mu_sensitivity_label
)

np.random.seed(SEED)
torch.manual_seed(SEED)

CONVERGENCE_THRESHOLD = 0.85
K_PARTICIPANTS = 3   # nœuds actifs par round (partial participation)
MU_VALUES = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]


# RUNNER PRINCIPAL

def run_experiment(X_train, y_train, X_test, y_test,
                   split_fn, mu=0.0, label="",
                   partial=False, k=None):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    test_loader = DataLoader(
        make_dataset(X_test, y_test), batch_size=256, shuffle=False)
    node_datasets = split_fn(X_train, y_train, N_NODES)
    node_loaders  = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
                     for ds in node_datasets]

    sizes = [len(ds) for ds in node_datasets]
    global_model = LightMLP()
    comm = communication_cost_per_round(global_model, k if partial else N_NODES)

    print(f"  Paramètres modèle     : {comm['params']:,}")
    print(f"  Participation/round   : {k if partial else N_NODES}/{N_NODES} nœuds"
          f"{'  ← topologie variable' if partial else ''}")
    print(f"  Coût comm./round      : {comm['KB']} KB")
    print(f"  Données par nœud      : {sizes}")

    history = {"accuracy": [], "loss": []}
    rng = np.random.RandomState(SEED)

    for rnd in range(1, N_ROUNDS + 1):
        # Sélection des nœuds participants
        if partial and k:
            participants = select_participants(N_NODES, k, rng)
        else:
            participants = list(range(N_NODES))

        active_loaders = [node_loaders[i] for i in participants]
        active_sizes   = [sizes[i] for i in participants]
        total_active   = sum(active_sizes)
        fed_weights    = [s / total_active for s in active_sizes]

        local_models = [
            local_train(
                copy.deepcopy(global_model), loader,
                epochs=LOCAL_EPOCHS, lr=LR,
                mu=mu, global_model=global_model
            )
            for loader in active_loaders
        ]

        global_model = federated_average(
            global_model, local_models, fed_weights)
        acc, loss = evaluate(global_model, test_loader)
        history["accuracy"].append(acc)
        history["loss"].append(loss)

        part_str = f"[nœuds {participants}]" if partial else ""
        print(f"  Round {rnd:2d}/{N_ROUNDS} | "
              f"Acc: {acc*100:.2f}% | Loss: {loss:.4f} {part_str}")

    rtc  = rounds_to_convergence(history["accuracy"], CONVERGENCE_THRESHOLD)
    active_n = k if partial else N_NODES
    cost = communication_cost_per_round(global_model, active_n)

    history["rounds_to_convergence"]  = rtc
    history["comm_cost_KB_per_round"] = cost["KB"]
    history["total_comm_cost_KB"]     = round(
        cost["KB"] * (rtc if rtc else N_ROUNDS), 1)
    history["n_participants"]         = active_n

    if rtc:
        print(f"\n  ✅ Seuil {int(CONVERGENCE_THRESHOLD*100)}% → round {rtc} "
              f"| Coût total : {history['total_comm_cost_KB']} KB")
    else:
        print(f"\n  ⚠️  Seuil {int(CONVERGENCE_THRESHOLD*100)}% non atteint")

    return history


# ANALYSE SENSIBILITÉ μ

def run_mu_sensitivity(X_train, y_train, X_test, y_test):
    print(f"\n{'='*65}")
    print(f"  Analyse de sensibilité μ — FedProx sur non-IID")
    print(f"  Valeurs testées : {MU_VALUES}")
    print(f"{'='*65}")

    test_loader = DataLoader(
        make_dataset(X_test, y_test), batch_size=256, shuffle=False)
    node_datasets = split_non_iid(X_train, y_train, N_NODES)
    node_loaders  = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
                     for ds in node_datasets]
    sizes = [len(ds) for ds in node_datasets]
    fed_weights = [s / sum(sizes) for s in sizes]

    mu_results = {}
    for mu in MU_VALUES:
        label = mu_sensitivity_label(mu)
        print(f"\n  → {label}")
        global_model = LightMLP()
        history = {"accuracy": [], "loss": []}

        for rnd in range(1, N_ROUNDS + 1):
            local_models = [
                local_train(copy.deepcopy(global_model), loader,
                            epochs=LOCAL_EPOCHS, lr=LR,
                            mu=mu, global_model=global_model)
                for loader in node_loaders
            ]
            global_model = federated_average(
                global_model, local_models, fed_weights)
            acc, loss = evaluate(global_model, test_loader)
            history["accuracy"].append(acc)
            history["loss"].append(loss)

        rtc = rounds_to_convergence(history["accuracy"], CONVERGENCE_THRESHOLD)
        history["rounds_to_convergence"] = rtc
        history["final_accuracy"] = history["accuracy"][-1]
        mu_results[str(mu)] = history

        print(f"     Acc finale : {history['final_accuracy']*100:.2f}% "
              f"| Rounds→85% : {rtc if rtc else 'N/A'}")

    return mu_results


# MAIN

def main():
    print("=" * 65)
    print("  FedEdge6G — Simulation DFL | Réseau 6G 4 Nœuds")
    print("=" * 65)
    print(f"\n  Classes de trafic : {CLASS_NAMES}")
    print("\n  Profils nœuds (non-IID) :")
    for k, v in NODE_PROFILES.items():
        print(f"    Nœud {k} : {v}")

    print("\n📊 Génération des données...")
    X, y = generate_traffic_data()
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print(f"   Train: {len(y_train)} | Test: {len(y_test)}")

    results = {}

    # Expérience 1 — FedAvg + IID
    results["FedAvg_IID"] = run_experiment(
        X_train, y_train, X_test, y_test,
        split_fn=split_iid, mu=0.0,
        label="Exp 1 — FedAvg + IID (baseline optimiste)")

    # Expérience 2 — FedAvg + non-IID
    results["FedAvg_nonIID"] = run_experiment(
        X_train, y_train, X_test, y_test,
        split_fn=split_non_iid, mu=0.0,
        label="Exp 2 — FedAvg + non-IID [Verrou 1a : données hétérogènes]")

    # Expérience 3 — FedProx + non-IID
    results["FedProx_nonIID"] = run_experiment(
        X_train, y_train, X_test, y_test,
        split_fn=split_non_iid, mu=MU_FEDPROX,
        label=f"Exp 3 — FedProx (μ={MU_FEDPROX}) + non-IID [Solution Verrou 1a]")

    # Expérience 4 — FedProx + non-IID + Partial Participation
    results["FedProx_nonIID_partial"] = run_experiment(
        X_train, y_train, X_test, y_test,
        split_fn=split_non_iid, mu=MU_FEDPROX,
        label=f"Exp 4 — FedProx + non-IID + Participation partielle ({K_PARTICIPANTS}/{N_NODES}) [Verrou 1b : topologies variables]",
        partial=True, k=K_PARTICIPANTS)

    # Expérience 5 — Sensibilité μ
    results["mu_sensitivity"] = run_mu_sensitivity(
        X_train, y_train, X_test, y_test)

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    #  Tableau synthèse 
    print("\n" + "=" * 72)
    print("  SYNTHÈSE FINALE")
    print("=" * 72)
    print(f"  {'Expérience':<38} {'Acc.':<10} {'→85%':<10} {'Coût'}")
    print("-" * 72)

    exp_labels = {
        "FedAvg_IID":            "FedAvg + IID",
        "FedAvg_nonIID":         "FedAvg + non-IID        [Verrou 1a]",
        "FedProx_nonIID":        "FedProx + non-IID       [Solution]",
        "FedProx_nonIID_partial":f"FedProx + non-IID + {K_PARTICIPANTS}/{N_NODES} nœuds [Verrou 1b]",
    }
    for key, lbl in exp_labels.items():
        h = results[key]
        acc  = h["accuracy"][-1] * 100
        rtc  = h.get("rounds_to_convergence")
        cost = h.get("total_comm_cost_KB", "N/A")
        print(f"  {lbl:<38} {acc:<10.2f}% "
              f"{'R'+str(rtc) if rtc else 'N/A':<10} {cost} KB")

    print("\n  Sensibilité μ (non-IID) :")
    for mu_val, h in results["mu_sensitivity"].items():
        acc = h["final_accuracy"] * 100
        rtc = h.get("rounds_to_convergence")
        print(f"    μ={float(mu_val):<5} → Acc: {acc:.2f}% "
              f"| →85% : {'R'+str(rtc) if rtc else 'N/A'}")

    print(f"\n  📁 Résultats → results/results.json")
    print(f"  🔬 Lance visualize.py pour le Pareto front et les graphiques")


if __name__ == "__main__":
    main()