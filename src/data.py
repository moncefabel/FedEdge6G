import numpy as np
import torch
from torch.utils.data import TensorDataset
from src.config import (
    N_CLASSES, N_FEATURES, N_SAMPLES, TEST_RATIO, SEED
)


def generate_traffic_data(n_samples=N_SAMPLES, seed=SEED):
    """
    Génère un dataset synthétique de trafic réseau 6G.

    Chaque classe a une "signature" de features distincte
    représentant ses caractéristiques réseau :
        [débit, latence, taille_paquets, fréquence,
         jitter, perte, priorité, bande_passante,
         délai_E2E, densité_devices, puissance_tx, QoS_score]

    Returns:
        X : np.array (n_samples, N_FEATURES), float32
        y : np.array (n_samples,), int64
    """
    rng = np.random.RandomState(seed)

    # Signature de chaque classe de trafic
    class_params = [
        # mean_features                                           std
        (np.array([0.1,0.9,0.2,0.3,0.8,0.1,0.5,0.2,0.3,0.1,0.4,0.2]), 0.15),  # VoIP
        (np.array([0.9,0.3,0.8,0.7,0.2,0.5,0.3,0.8,0.6,0.7,0.2,0.6]), 0.12),  # Video
        (np.array([0.1,0.2,0.1,0.9,0.1,0.8,0.9,0.1,0.2,0.9,0.8,0.1]), 0.10),  # IoT
        (np.array([0.8,0.4,0.9,0.5,0.3,0.3,0.4,0.9,0.7,0.4,0.3,0.8]), 0.13),  # eMBB
        (np.array([0.3,0.95,0.3,0.2,0.9,0.2,0.2,0.3,0.95,0.2,0.1,0.3]), 0.08),# URLLC
        (np.array([0.05,0.1,0.05,0.8,0.05,0.9,0.8,0.05,0.1,0.8,0.9,0.05]),0.09),# mMTC
    ]

    X, y = [], []
    per_class = n_samples // N_CLASSES
    for c, (mean, std) in enumerate(class_params):
        features = rng.normal(mean, std, (per_class, N_FEATURES))
        features = np.clip(features, 0, 1)
        X.append(features)
        y.extend([c] * per_class)

    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def train_test_split(X, y, test_ratio=TEST_RATIO, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    return (X[idx[n_test:]], y[idx[n_test:]],
            X[idx[:n_test]],  y[idx[:n_test]])


def make_dataset(X, y):
    return TensorDataset(torch.tensor(X), torch.tensor(y))


def split_iid(X_train, y_train, n_nodes, seed=SEED):
    """
    Partition IID : chaque nœud reçoit une portion
    aléatoire et uniforme de toutes les classes.

    → Hypothèse optimiste et irréaliste.
      Sert de référence upper-bound dans nos expériences.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y_train))
    size = len(y_train) // n_nodes
    return [
        make_dataset(X_train[idx[i*size:(i+1)*size]],
                     y_train[idx[i*size:(i+1)*size]])
        for i in range(n_nodes)
    ]


def split_non_iid(X_train, y_train, n_nodes, seed=SEED):
    """
    Partition non-IID : chaque nœud a un profil de trafic dominant.

    Distribution par nœud :
        Nœud 0 (Résidentiel)   : 70% VoIP + Video Streaming
        Nœud 1 (Industriel)    : 70% IoT Industriel + URLLC
        Nœud 2 (Dense urbain)  : 70% eMBB + mMTC
        Nœud 3 (Mixte)         : distribution uniforme

    → Simule la réalité des déploiements 6G hétérogènes.
      C'est le scénario central du Verrou 1.
    """
    rng = np.random.RandomState(seed)
    dominant = [[0, 1], [2, 4], [3, 5], None]
    class_idx = {c: np.where(y_train == c)[0] for c in range(N_CLASSES)}

    datasets = []
    for i in range(n_nodes):
        if dominant[i] is None:
            idx = rng.permutation(len(y_train))[:1500]
        else:
            dom, other = dominant[i], [c for c in range(N_CLASSES) if c not in dominant[i]]
            idx = []
            for c in dom:
                idx.extend(rng.permutation(class_idx[c])[:530].tolist())
            for c in other:
                idx.extend(rng.permutation(class_idx[c])[:60].tolist())
            idx = np.array(idx)
            rng.shuffle(idx)
        datasets.append(make_dataset(X_train[idx], y_train[idx]))

    return datasets