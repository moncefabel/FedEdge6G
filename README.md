# FedEdge6G

**Federated Learning Simulation for Heterogeneous 6G Edge Networks**

Simulation of Distributed Federated Learning (DFL) on a 4-node 6G network topology, comparing **FedAvg** and **FedProx** under **IID** and **non-IID** data distributions.

This project is part of my preparatory research for the CIFRE PhD thesis *"Orchestration dynamique d'IA distribuée pour des réseaux 6G économes en énergie"* at Orange Innovation (ref. 2026-51929) — Project TREES.

---

## Motivation

In real 6G deployments, each edge node serves a geographically distinct zone and observes a **dominant traffic type**:

| Node | Profile | Dominant Traffic |
|------|---------|-----------------|
| Node 0 | Residential | VoIP + Video Streaming (70%) |
| Node 1 | Industrial | IoT + URLLC (70%) |
| Node 2 | Dense Urban | eMBB + mMTC (70%) |
| Node 3 | Mixed | Uniform distribution |

This heterogeneity creates a **non-IID data distribution** across nodes — one of the core scientific challenges (*Verrou 1*) of the TREES thesis: nodes training on different distributions cause local models to diverge, degrading global convergence.

**Research question:** Can FedProx's proximal regularization term mitigate this client drift under realistic non-IID 6G conditions?

---

## Scientific Locks Addressed

This simulation directly addresses two of the four scientific locks identified in the thesis:

**Verrou 1 — Dynamic placement under non-IID data and variable topologies**
> Standard federated algorithms assume IID data across nodes. In real 6G deployments, distributions are heterogeneous and dynamic. This simulation quantifies the performance gap and evaluates FedProx as a mitigation strategy.

**Verrou 3 — Frugal solution for energy efficiency while preserving QoS**
> The model architecture (LightMLP, ~10K parameters) is deliberately constrained to simulate a compute-limited edge node, consistent with Orange's carbon neutrality objective for 2040.

---

## Algorithms

### FedAvg (McMahan et al., 2017)
Standard federated averaging. Each node trains locally, then the server aggregates weights by weighted average. Under non-IID data, local models drift in opposite directions (*client drift*), slowing convergence.

### FedProx (Li et al., 2020)
Adds a proximal term to the local loss:

```
L_local(w) = L(w) + (μ/2) · ‖w − w_global‖²
```

This penalizes deviation from the global model, limiting client drift. Particularly effective under heterogeneous (non-IID) data distributions.

---

## Dataset

Synthetic 6G network traffic dataset with **6 traffic classes** and **12 network features** (throughput, latency, packet size, jitter, QoS score, etc.):

| Class | Type | Characteristics |
|-------|------|----------------|
| 0 | VoIP | Low throughput, latency-critical |
| 1 | Video Streaming | High throughput, latency-tolerant |
| 2 | IoT Industriel | Small packets, high frequency |
| 3 | eMBB | Very high throughput |
| 4 | URLLC | Ultra-low latency, reliability-critical |
| 5 | mMTC | Massive devices, low power |

9,000 samples generated with class-specific feature distributions. 80/20 train/test split.

---

## Results

| Experiment | Final Accuracy (Round 20) |
|-----------|--------------------------|
| FedAvg + IID | 85.83% |
| FedAvg + non-IID *(realistic 6G)* | 86.44% |
| **FedProx + non-IID** | **92.72%** ✅ |
| FedProx + IID | 85.22% |

**Key finding:** FedProx recovers **+6.3%** over FedAvg under non-IID distribution, empirically validating Verrou 1: heterogeneous data degrades DFL convergence, and the proximal regularization term effectively mitigates client drift.

![DFL Results](results/dfl_results.png)

---

## Project Structure

```
FedEdge6G/
├── src/
│   ├── config.py        # Centralized hyperparameters
│   ├── model.py         # LightMLP architecture (frugal edge model)
│   ├── data.py          # Traffic data generation + IID/non-IID splits
│   └── federation.py    # FedAvg, FedProx, local training, evaluation
├── experiments/
│   ├── run_all.py       # Run all 4 experiments
│   └── visualize.py     # Generate result plots
├── results/             # Output (JSON + PNG)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments (20 rounds × 4 configs)
python -m experiments.run_all

# Generate visualizations
python -m experiments.visualize
```

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Nodes (N) | 4 |
| Communication rounds | 20 |
| Local epochs per round | 3 |
| Batch size | 64 |
| Learning rate | 0.001 |
| FedProx μ | 0.1 |

---

## References

- McMahan, H. B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS. [FedAvg]
- Li, T., et al. (2020). *Federated Optimization in Heterogeneous Networks.* MLSys. [FedProx]
- Latreche, A., & Bellahsene, Z. (2026). *A Comprehensive Survey on 6G.* Franklin Open.
- Chatzieleftheriou, L., & Liotou, E. (2026). *A Survey on AI for 6G.* IEEE Open Journal of Communications.

---

## Context

This project was implemented as part of the preparatory research for the CIFRE PhD thesis at **Orange Innovation** (ref. 2026-51929), within the ANR collaborative project **TREES** (*Towards Resilient and Energy-Efficient 6G Systems*), in partnership with Université d'Avignon, Paris Dauphine, and CNAM Paris.

**Author:** Moncef Bouhabel — Machine Learning Engineer  
**Supervisors (thesis):** Guillaume Fraysse (Orange Innovation)  
**Contact:** moncef.bmd@gmail.com