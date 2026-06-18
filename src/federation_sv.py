import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.federation import federated_average, evaluate


def compute_loo_sv(global_model, local_models, test_loader):
    """
    Leave-One-Out approximation des Shapley Values.

    Pour chaque nœud i :
        1. Agréger tous les modèles locaux → acc_full
        2. Agréger tous sauf i → acc_loo_i
        3. SV_i ≈ acc_full - acc_loo_i

    Complexité : O(n) agrégations par round (vs O(2^n) pour Shapley exact).
    Approximation valide sous hypothèse de symétrie des contributions.

    Args:
        global_model  : modèle global du round précédent
        local_models  : liste des modèles locaux entraînés ce round
        test_loader   : DataLoader du jeu de test global

    Returns:
        sv_scores (list[float]) : score SV par nœud
        acc_full  (float)       : accuracy du modèle agrégé complet
        sv_weights (list[float]): poids normalisés (ReLU sur SV)
    """
    n = len(local_models)

    # 1. Modèle agrégé avec tous les nœuds
    model_full = copy.deepcopy(global_model)
    model_full = federated_average(model_full, local_models)
    acc_full, _ = evaluate(model_full, test_loader)

    # 2. Leave-one-out : agréger sans chaque nœud i
    sv_scores = []
    for i in range(n):
        models_loo = [m for j, m in enumerate(local_models) if j != i]
        model_loo = copy.deepcopy(global_model)
        model_loo = federated_average(model_loo, models_loo)
        acc_loo, _ = evaluate(model_loo, test_loader)
        sv_i = acc_full - acc_loo
        sv_scores.append(sv_i)

    # 3. Poids : ReLU(SV) normalisé
    weights_raw = np.array([max(0.0, sv) for sv in sv_scores])
    if weights_raw.sum() < 1e-8:
        # Fallback : poids uniformes
        sv_weights = [1.0 / n] * n
    else:
        sv_weights = (weights_raw / weights_raw.sum()).tolist()

    return sv_scores, acc_full, sv_weights


def federated_average_sv(global_model, local_models, test_loader):
    """
    Agrégation FedSV : pondération par Shapley Values leave-one-out.

    Les nœuds avec SV ≤ 0 (contribution nulle ou négative) sont exclus.
    Les nœuds avec SV > 0 sont pondérés proportionnellement.

    Returns:
        global_model  : modèle agrégé pondéré par SV
        sv_scores     : scores SV bruts par nœud
        sv_weights    : poids finaux utilisés pour l'agrégation
    """
    sv_scores, acc_full, sv_weights = compute_loo_sv(
        global_model, local_models, test_loader
    )
    global_model = federated_average(global_model, local_models, sv_weights)
    return global_model, sv_scores, sv_weights