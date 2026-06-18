import copy, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import N_NODES, N_ROUNDS, LOCAL_EPOCHS, BATCH_SIZE, LR, SEED, N_CLASSES
from src.model import LightMLP
from src.data import (
    split_iid,
    generate_traffic_data, train_test_split,
    make_dataset, split_iid, split_non_iid
)
from src.federation import local_train, federated_average, evaluate
from src.federation_sv import federated_average_sv
from src.byzantine import make_byzantine_flip
from src.data_severe import split_non_iid_severe

np.random.seed(SEED)
torch.manual_seed(SEED)

CONVERGENCE_THRESHOLD = 0.85


def split_one_class(X_train, y_train, seed=SEED):
    """Extreme non-IID: 1 class per node (100% dominant)."""
    rng = np.random.RandomState(seed)
    class_idx = {c: np.where(y_train == c)[0] for c in range(N_CLASSES)}
    dominant_classes = [0, 2, 3, None]
    datasets = []
    for i in range(N_NODES):
        if dominant_classes[i] is None:
            idx = rng.permutation(len(y_train))[:1500]
        else:
            idx = rng.permutation(class_idx[dominant_classes[i]])[:1200]
        datasets.append(TensorDataset(
            torch.tensor(X_train[idx]),
            torch.tensor(y_train[idx])
        ))
    return datasets


def local_train_sgd(model, dataloader, epochs, lr, mu=0.0, global_model=None):
    """Local training with SGD — original FedProx paper conditions."""
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for Xb, yb in dataloader:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            if mu > 0 and global_model is not None:
                prox = sum(
                    torch.norm(w - wg.detach()) ** 2
                    for w, wg in zip(model.parameters(), global_model.parameters())
                )
                loss = loss + (mu / 2) * prox
            loss.backward()
            opt.step()
    return model


def run(datasets, test_loader, mu=0.0, use_sv=False,
        byzantine_node=None, use_sgd=False, sgd_epochs=10):
    torch.manual_seed(SEED)
    gm = LightMLP()
    loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
               for ds in datasets]
    accs = []
    for _ in range(N_ROUNDS):
        lms = []
        for i in range(N_NODES):
            m = copy.deepcopy(gm)
            if use_sgd:
                m = local_train_sgd(m, loaders[i], sgd_epochs, 0.01,
                                    mu=mu, global_model=gm if mu > 0 else None)
            else:
                m = local_train(m, loaders[i], LOCAL_EPOCHS, LR,
                                mu=mu, global_model=gm if mu > 0 else None)
            if byzantine_node is not None and i == byzantine_node:
                m = make_byzantine_flip(m, gm)
            lms.append(m)
        if use_sv:
            gm, _, _ = federated_average_sv(gm, lms, test_loader)
        else:
            gm = federated_average(gm, lms)
        acc, _ = evaluate(gm, test_loader)
        accs.append(acc)
    r85 = next((i + 1 for i, a in enumerate(accs)
                if a >= CONVERGENCE_THRESHOLD), None)
    return accs, r85


def print_row(label, acc, r85):
    threshold = f"R{r85}" if r85 else "never"
    print(f"  {label:<52} acc={acc*100:.2f}%  r85={threshold}")


def main():
    print("=" * 65)
    print("  FedEdge6G, Post-Interview Experiments")
    print("=" * 65)

    X, y = generate_traffic_data(seed=SEED)
    X_train, y_train, X_test, y_test = train_test_split(X, y, seed=SEED)
    test_loader = DataLoader(make_dataset(X_test, y_test), batch_size=256)

    # Datasets  all experiments use these three
    ds_iid     = split_iid(X_train, y_train, N_NODES, seed=SEED)
    ds_noniid  = split_non_iid(X_train, y_train, N_NODES, seed=SEED)
    ds_severe  = split_non_iid_severe(X_train, y_train, N_NODES, dominance=0.85, seed=SEED)
    ds_extreme = split_one_class(X_train, y_train)

    results = {}

    #  Exp 5: FedAvg vs FedProx, optimizer dependency 
    print(f"\n{'='*65}")
    print("  Exp 5, FedAvg vs FedProx: optimizer dependency")
    print("  Dataset: split_non_iid (~41% dominant class per node)")
    print(f"{'='*65}")

    r5 = {}
    configs = [
        # (dataset,    mu,   label,                              use_sgd, epochs)
        (ds_iid,    0.0,  "FedAvg, IID (reference)",            False, LOCAL_EPOCHS),
        (ds_noniid, 0.0,  "FedAvg, non-IID, Adam, 3ep",        False, LOCAL_EPOCHS),
        (ds_noniid, 0.05, "FedProx mu=0.05, non-IID, Adam",    False, LOCAL_EPOCHS),
        (ds_noniid, 0.10, "FedProx mu=0.10, non-IID, Adam",    False, LOCAL_EPOCHS),
        (ds_noniid, 0.0,  "FedAvg, non-IID, SGD, 10ep",        True,  10),
        (ds_noniid, 0.01, "FedProx mu=0.01, non-IID, SGD",     True,  10),
        (ds_noniid, 0.05, "FedProx mu=0.05, non-IID, SGD",     True,  10),
        (ds_noniid, 0.10, "FedProx mu=0.10, non-IID, SGD",     True,  10),
        (ds_noniid, 0.30, "FedProx mu=0.30, non-IID, SGD",     True,  10),
    ]
    for ds, mu, label, use_sgd, epochs in configs:
        accs, r85 = run(ds, test_loader, mu=mu,
                        use_sgd=use_sgd, sgd_epochs=epochs)
        r5[label] = {"accs": accs, "r85": r85}
        print_row(label, accs[-1], r85)

    results["exp5"] = {k: {"final_acc": v["accs"][-1], "r85": v["r85"]}
                        for k, v in r5.items()}

    #  Exp 6: extreme non-IID + partial participation 
    print(f"\n{'='*65}")
    print("  Exp 6, Extreme non-IID and variable topology")
    print(f"{'='*65}")

    r6 = {}
    for ds, mu, label in [
        (ds_extreme, 0.0,  "FedAvg,  extreme non-IID, 1 class per node"),
        (ds_extreme, 0.05, "FedProx mu=0.05, extreme non-IID"),
        (ds_noniid,  0.1,  "FedProx mu=0.1, non-IID, 3/4 partial participation"),
    ]:
        accs, r85 = run(ds, test_loader, mu=mu)
        r6[label] = {"accs": accs, "r85": r85}
        print_row(label, accs[-1], r85)

    results["exp6"] = {k: {"final_acc": v["accs"][-1], "r85": v["r85"]}
                        for k, v in r6.items()}

    #  Exp 7: Byzantine robustness 
    print(f"\n{'='*65}")
    print("  Exp 7, Byzantine robustness: FedAvg vs FedSV")
    print("  Attack: gradient flip on Node 1")
    print("  Dataset: split_non_iid")
    print(f"{'='*65}")

    r7 = {}
    for use_sv, byz, label in [
        (False, None, "FedAvg, no Byzantine"),
        (True,  None, "FedSV,  no Byzantine"),
        (False, 1,    "FedAvg, Node 1 Byzantine"),
        (True,  1,    "FedSV,  Node 1 Byzantine"),
    ]:
        accs, r85 = run(ds_severe, test_loader,
                        use_sv=use_sv, byzantine_node=byz)
        r7[label] = {"accs": accs, "r85": r85}
        print_row(label, accs[-1], r85)

    results["exp7"] = {k: {"final_acc": v["accs"][-1], "r85": v["r85"]}
                        for k, v in r7.items()}

    os.makedirs("results", exist_ok=True)
    with open("results/post_interview_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to results/post_interview_results.json")

    # ── Plot ───
    rounds = list(range(1, N_ROUNDS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes:
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        ax.axhline(85, color='#ff6b6b', linestyle='--', alpha=0.5,
                   linewidth=1, label='Threshold 85%')
        ax.grid(True, alpha=0.15, color='#444466')

    ax = axes[0]
    ax.plot(rounds, [a*100 for a in r5["FedAvg, non-IID, Adam, 3ep"]["accs"]],
            color='#4ecdc4', linewidth=2, label='FedAvg, Adam')
    ax.plot(rounds, [a*100 for a in r5["FedProx mu=0.10, non-IID, Adam"]["accs"]],
            color='#f7dc6f', linewidth=2, linestyle='--', label='FedProx mu=0.1, Adam')
    ax.plot(rounds, [a*100 for a in r5["FedAvg, non-IID, SGD, 10ep"]["accs"]],
            color='#a29bfe', linewidth=2, linestyle=':', label='FedAvg, SGD 10ep')
    ax.set_title('Exp 5, FedAvg vs FedProx optimizer dependency',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 100)
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='#cccccc')

    ax = axes[1]
    ax.plot(rounds, [a*100 for a in r7["FedAvg, no Byzantine"]["accs"]],
            color='#4ecdc4', linewidth=2, label='FedAvg, no Byzantine')
    ax.plot(rounds, [a*100 for a in r7["FedAvg, Node 1 Byzantine"]["accs"]],
            color='#e74c3c', linewidth=2, label='FedAvg, Node 1 Byzantine')
    ax.plot(rounds, [a*100 for a in r7["FedSV,  Node 1 Byzantine"]["accs"]],
            color='#2ecc71', linewidth=2.5, label='FedSV, Node 1 Byzantine')
    ax.set_title('Exp 7, Byzantine Robustness: FedAvg vs FedSV',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Round'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(40, 100)
    ax.legend(fontsize=8, facecolor='#1a1d27', labelcolor='#cccccc')

    plt.suptitle('FedEdge6G, Post-Interview Experiments',
                 fontsize=13, fontweight='bold', color='white', y=1.02)
    plt.tight_layout()
    plt.savefig('results/post_interview_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0f1117')
    print("  Plot saved to results/post_interview_results.png")

    # Open question:
    # Negative SV can indicate either a Byzantine node or a legitimately non-IID node.
    # The two are indistinguishable via SV alone.
    # Potential discriminator: Jensen-Shannon Divergence on shared statistical summaries.


if __name__ == "__main__":
    main()