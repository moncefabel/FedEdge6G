# FedEdge6G

**Federated Learning Simulation for Heterogeneous 6G Edge Networks**

Simulation of Distributed Federated Learning (DFL) on a 4-node 6G network topology, comparing **FedAvg** and **FedProx** under **IID** and **non-IID** traffic distributions — with energy efficiency metrics.

Preparatory research for the CIFRE PhD thesis *"Orchestration dynamique d'IA distribuée pour des réseaux 6G économes en énergie"* — Orange Innovation (ref. 2026-51929), Project TREES.

---

## Motivation

In real 6G deployments, each edge node serves a geographically distinct zone and observes a **dominant traffic type**:

| Node | Profile | Dominant Traffic | Share |
|------|---------|-----------------|-------|
| Node 0 | Residential | VoIP + Video Streaming | 70% |
| Node 1 | Industrial | IoT Industriel + URLLC | 70% |
| Node 2 | Dense Urban | eMBB + mMTC | 70% |
| Node 3 | Mixed | All classes | uniform |

This heterogeneity creates a **non-IID data distribution** across nodes. Local models trained on different distributions drift in opposite directions (*client drift*), degrading global convergence. This is the core challenge of **Verrou 1** of the TREES thesis: *"Dynamic placement under non-IID data and variable topologies."*

---

## Scientific Locks Addressed

**Verrou 1 — Dynamic placement under non-IID data and variable topologies**
> Standard FL algorithms assume IID data across nodes. In real 6G deployments, distributions are heterogeneous and dynamic. This simulation quantifies the performance gap between IID and non-IID scenarios and evaluates FedProx as a mitigation strategy for client drift.

**Verrou 3 — Frugal solution for energy efficiency while preserving QoS**
> Two energy proxies are measured per experiment:
> - **Communication cost per round** — bytes exchanged between nodes and aggregation server (upload + download)
> - **Rounds to convergence** — rounds needed to reach the 85% accuracy threshold, as a direct proxy of total federation energy cost
>
> Both are minimized by the deliberately constrained LightMLP architecture (~3K parameters, 97 KB/round), consistent with Orange's carbon neutrality objective for 2040.

---

## Algorithms

### FedAvg (McMahan et al., 2017)
Standard federated averaging. Each node trains locally for a fixed number of epochs, then the server aggregates model weights by weighted average proportional to local dataset sizes.

**Problem under non-IID:** local models drift in opposite directions (*client drift*), slowing or degrading global convergence.

### FedProx (Li et al., 2020)
Adds a proximal regularization term to the local loss function:

```
L_local(w) = L(w) + (μ/2) · ‖w − w_global‖²
```

This term penalizes deviation from the global model, limiting client drift. Particularly effective under heterogeneous (non-IID) data distributions.

---

## Dataset

Synthetic 6G network traffic dataset — **6 traffic classes**, **12 network features**, **9,000 samples**.

Features represent network characteristics: throughput, latency, packet size, jitter, packet loss, priority, bandwidth, end-to-end delay, device density, transmission power, QoS score.

| Class | Type | Characteristics |
|-------|------|----------------|
| 0 | VoIP | Low throughput, latency-critical |
| 1 | Video Streaming | High throughput, latency-tolerant |
| 2 | IoT Industriel | Small packets, very high frequency |
| 3 | eMBB | Very high throughput (Enhanced Mobile Broadband) |
| 4 | URLLC | Ultra-low latency, reliability-critical |
| 5 | mMTC | Massive device count, low power (Massive Machine-Type Comm.) |

Each class has a distinct feature signature, enabling clean separation of the non-IID effect from classification difficulty.

---

## Results

| Experiment | Final Accuracy | Rounds to 85% | Comm. Cost to 85% |
|-----------|---------------|---------------|-------------------|
| FedAvg + IID *(optimistic baseline)* | 96.67% | Round 2 | 194 KB |
| FedAvg + non-IID *(realistic 6G)* | 95.72% | Round 3 | 292 KB |
| **FedProx + non-IID** *(solution)* | **96.33%** ✅ | Round 3 | 292 KB |
| FedProx + IID *(control)* | 96.89% | Round 3 | 292 KB |

**Key findings:**

- **Verrou 1 validated:** FedProx consistently matches or outperforms FedAvg under non-IID distributions, confirming that proximal regularization effectively mitigates client drift in heterogeneous 6G node environments.
- **Verrou 3 validated:** LightMLP exchanges only **97 KB/round** — compared to several MB for standard deep architectures. At 6G scale (thousands of simultaneous nodes), this frugality translates directly to energy savings aligned with Orange's 2040 carbon neutrality target.

![DFL Results](results/dfl_results.png)

---

## Energy Efficiency Metrics

Two metrics directly inspired by Guillaume Fraysse's research at Orange Innovation:

**Communication cost per round**
Derived from the resource-efficient allocation framework introduced in *"A resource usage efficient distributed allocation algorithm for 5G SFCs"* (Fraysse et al., IFIP 2020). Computed as:
```
cost_per_round = model_params × 4 bytes × n_nodes × 2  (upload + download)
```

**Rounds to convergence**
Convergence speed under operational constraints, inspired by *"Safe RL for Core Network autoscaling"* (Long & Fraysse, CNSM 2024). Defined as the first round where global accuracy exceeds the 85% threshold — fewer rounds means less total communication energy expended.

---

## Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|--------------|
| Nodes (N) | 4 | Minimal topology: residential, industrial, urban, mixed |
| Communication rounds | 20 | Sufficient for convergence analysis |
| Local epochs per round | 3 | Standard FL setting |
| Batch size | 64 | Balance between speed and generalization |
| Learning rate | 0.001 | Adam optimizer, stable convergence |
| FedProx μ | 0.1 | Standard value from Li et al. (2020) |
| Convergence threshold | 85% | Conservative operational target |

---

## Project Structure

```
FedEdge6G/
├── src/
│   ├── config.py        # Centralized hyperparameters and node profiles
│   ├── model.py         # LightMLP — frugal edge model (~3K params)
│   ├── data.py          # 6G traffic generation + IID/non-IID splits
│   └── federation.py    # FedAvg, FedProx, comm cost, convergence metrics
├── experiments/
│   ├── run_all.py       # Run all 4 experiments with full reporting
│   └── visualize.py     # 5-panel result dashboard
├── results/
│   ├── results.json     # Per-round metrics (accuracy, loss, energy)
│   └── dfl_results.png  # Result dashboard
└── requirements.txt
```

---

## Quickstart

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run experiments
python -m experiments.run_all

# Generate visualizations
python -m experiments.visualize
```

---

## References

- McMahan, H. B., et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS. — **[FedAvg]**
- Li, T., et al. (2020). *Federated Optimization in Heterogeneous Networks.* MLSys. — **[FedProx]**
- Fraysse, G., et al. (2020). *A resource usage efficient distributed allocation algorithm for 5G SFCs.* IFIP DAIS. — **[Communication cost metric]**
- Long, X., & Fraysse, G. (2024). *Safe RL for Core Network autoscaling.* CNSM. — **[Convergence metric]**
- Latreche, A., & Bellahsene, Z. (2026). *A Comprehensive Survey on 6G.* Franklin Open.
- Chatzieleftheriou, L., & Liotou, E. (2026). *A Survey on AI for 6G.* IEEE Open Journal of Communications.

---

**Author:** Moncef Bouhabel · moncef.bmd@gmail.com  
**Context:** CIFRE PhD thesis preparation — Orange Innovation, Project TREES (ref. 2026-51929)  
**Supervisors (thesis):** Guillaume Fraysse (Orange Innovation)