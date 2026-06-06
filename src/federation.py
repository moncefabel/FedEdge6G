import torch
import torch.nn as nn
import torch.optim as optim
import copy


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

    Returns:
        model entraîné localement
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)

            # Terme proximal FedProx
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

    weights : proportions de données de chaque nœud
              (nœud avec plus de données → poids plus élevé)
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