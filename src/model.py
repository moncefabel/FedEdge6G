import torch.nn as nn
from src.config import N_FEATURES, N_CLASSES


class LightMLP(nn.Module):
    """
    MLP frugal pour classification de trafic réseau 6G.

    Architecture :
        Input(12) → Linear(64) → ReLU → Dropout(0.2)
                  → Linear(32) → ReLU → Linear(6)

    Paramètres : ~10 400
    Justification : adapté aux contraintes énergétiques
    des nœuds edge dans un réseau 6G distribué.
    """
    def __init__(self):
        super(LightMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)