import numpy as np
from torch.utils.data import TensorDataset
import torch
from src.config import N_CLASSES, SEED


def split_non_iid_severe(X_train, y_train, n_nodes, dominance=0.85, seed=SEED):
    """
    Partition avec dominance forte (défaut 85%).
    
    Nœud 0 : 85% VoIP + Video  / 15% reste
    Nœud 1 : 85% IoT + URLLC  / 15% reste
    Nœud 2 : 85% eMBB + mMTC  / 15% reste
    Nœud 3 : distribution uniforme
    """
    rng = np.random.RandomState(seed)
    dominant = [[0, 1], [2, 4], [3, 5], None]
    class_idx = {c: np.where(y_train == c)[0] for c in range(N_CLASSES)}
    total_per_node = 1300

    datasets = []
    for i in range(n_nodes):
        if dominant[i] is None:
            idx = rng.permutation(len(y_train))[:total_per_node]
        else:
            n_dominant = int(total_per_node * dominance)
            n_other    = total_per_node - n_dominant
            dom   = dominant[i]
            other = [c for c in range(N_CLASSES) if c not in dom]

            idx = []
            per_dom   = n_dominant // len(dom)
            per_other = n_other    // len(other)
            for c in dom:
                idx.extend(rng.permutation(class_idx[c])[:per_dom].tolist())
            for c in other:
                idx.extend(rng.permutation(class_idx[c])[:per_other].tolist())
            idx = np.array(idx)
            rng.shuffle(idx)

        X_node = torch.tensor(X_train[idx])
        y_node = torch.tensor(y_train[idx])
        datasets.append(TensorDataset(X_node, y_node))

    return datasets


def compute_entropy(dataset):
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    total  = len(labels)
    probs  = np.array([labels.count(c) / total for c in range(N_CLASSES)])
    return -np.sum(probs * np.log(probs + 1e-9))


def show_distribution(datasets, label):
    print(f"\n=== Distribution non-IID : {label} ===")
    for i, ds in enumerate(datasets):
        labels = [int(ds[j][1]) for j in range(len(ds))]
        total  = len(labels)
        dominant_class = max(range(N_CLASSES), key=lambda c: labels.count(c))
        dom_pct = labels.count(dominant_class) / total * 100
        entropy = compute_entropy(ds)
        print(f"  Node {i}: {total} samples | dominant={dominant_class} ({dom_pct:.0f}%) | H={entropy:.3f} bits")