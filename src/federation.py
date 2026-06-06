import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


def local_train(model, dataloader, epochs, lr, mu=0.0, global_model=None):
    """
    Entraînement local sur un nœud edge.

    Args:
        model        : modèle local (copie du modèle global)
        dataloader   : données locales du nœud
        epochs       : nombre d'epochs locaux
        lr           : learning rate
        mu           : coefficient proximal (0 = FedAvg, >0 = FedProx)
        global_model : référence au modèle global (pour terme proximal)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)

            if mu > 0 and global_model is not None:
                proximal_term = sum(
                    torch.norm(w - w_g.detach()) ** 2
                    for w, w_g in zip(model.parameters(),
                                      global_model.parameters())
                )
                loss = loss + (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

    return model


def federated_average(global_model, local_models, weights=None):
    """
    Agrégation FedAvg : moyenne pondérée des poids locaux.
    """
    if weights is None:
        weights = [1.0 / len(local_models)] * len(local_models)

    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = sum(
            w * m.state_dict()[key].float()
            for w, m in zip(weights, local_models)
        )
    global_model.load_state_dict(global_dict)
    return global_model


def evaluate(model, test_loader):
    """Évaluation du modèle global sur le jeu de test."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            total_loss += criterion(output, y_batch).item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total, total_loss / len(test_loader)


# MÉTRIQUES D'EFFICACITÉ ÉNERGÉTIQUE

def communication_cost_per_round(model, n_nodes):
    """
    Calcule le coût de communication d'un round de fédération.

    À chaque round, chaque nœud :
      - upload   son modèle local vers le serveur  (+)
      - download le modèle global agrégé           (+)
    Soit 2 × taille_modèle × n_nodes bytes échangés.

    Inspiré du travail de Fraysse et al. sur l'allocation
    efficace de ressources dans les réseaux 5G (IFIP 2020).

    Returns:
        dict avec bytes total, KB, et MB
    """
    # float32 = 4 bytes par paramètre
    params = sum(p.numel() for p in model.parameters())
    bytes_per_node = params * 4          # upload du nœud
    total_bytes = bytes_per_node * n_nodes * 2  # upload + download

    return {
        "params":      params,
        "bytes":       total_bytes,
        "KB":          round(total_bytes / 1024, 2),
        "MB":          round(total_bytes / (1024 ** 2), 4),
    }


def rounds_to_convergence(accuracy_history, threshold=0.85):
    """
    Retourne le premier round où l'accuracy dépasse le seuil.

    Proxy de la consommation énergétique totale :
    moins de rounds = moins d'échanges réseau = moins d'énergie.
    Cohérent avec l'objectif neutralité carbone 2040 d'Orange
    (Verrou 3 : arbitrage QoS / empreinte carbone).

    Inspiré de la métrique de convergence sous contraintes de
    "Safe RL for Core Network autoscaling" (Long & Fraysse, 2024).

    Args:
        accuracy_history : liste d'accuracy par round [0.0, 1.0]
        threshold        : seuil cible (défaut 0.85 = 85%)

    Returns:
        int  : numéro du round (1-indexed), ou None si non atteint
    """
    for i, acc in enumerate(accuracy_history):
        if acc >= threshold:
            return i + 1
    return None


# TOPOLOGIES VARIABLES — PARTICIPATION PARTIELLE

def select_participants(n_nodes, k, rng=None):
    """
    Sélectionne k nœuds parmi n_nodes pour un round donné.

    Simule les topologies variables du Verrou 1 :
    dans un réseau 6G réel, tous les nœuds ne sont pas
    disponibles simultanément (pannes, surcharge, mobilité).

    Inspiré du paradigme d'autoscaling dynamique de :
    "Autoscaling Packet Core Network Functions with Deep RL"
    (Singh, Verma, Matsuo, Fossati, Fraysse — NOMS 2023)

    Args:
        n_nodes : nombre total de nœuds
        k       : nombre de participants par round
        rng     : RandomState pour reproductibilité

    Returns:
        list[int] : indices des nœuds sélectionnés
    """
    if rng is None:
        rng = np.random.RandomState()
    return rng.choice(n_nodes, k, replace=False).tolist()


# ANALYSE μ — SENSIBILITÉ DU TERME PROXIMAL FEDPROX

def mu_sensitivity_label(mu):
    """Retourne une description du rôle de μ pour un affichage clair."""
    if mu == 0:
        return "FedAvg (μ=0)"
    elif mu < 0.05:
        return f"FedProx faible (μ={mu}) — contrainte légère"
    elif mu <= 0.2:
        return f"FedProx standard (μ={mu}) — contrainte modérée"
    else:
        return f"FedProx fort (μ={mu}) — contrainte forte"