# Topologie réseau
N_NODES = 4
N_ROUNDS = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LR = 0.001
MU_FEDPROX = 0.1   # Coefficient proximal FedProx (μ)

# Dataset
N_SAMPLES = 9000
TEST_RATIO = 0.2
N_FEATURES = 12
N_CLASSES = 6
SEED = 42

# Classes de trafic 6G simulées
CLASS_NAMES = [
    "VoIP",
    "Video Streaming",
    "IoT Industriel",
    "eMBB",
    "URLLC",
    "mMTC",
]

# Profils des nœuds en non-IID
NODE_PROFILES = {
    0: "Résidentiel  — VoIP + Video dominant (70%)",
    1: "Industriel   — IoT + URLLC dominant  (70%)",
    2: "Dense urbain — eMBB + mMTC dominant  (70%)",
    3: "Mixte        — distribution uniforme",
}