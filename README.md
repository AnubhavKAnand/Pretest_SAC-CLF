# HSW-SF Pretest Simulation
## Hierarchical Small-World SplitFed – Single-Machine Pretest

Simulates the system from the paper:
> *"Robust Federated Deep Reinforcement Learning for Massive UAV Swarms
>  via Hierarchical Small-World SplitFed Architecture"*

---

### File Overview

| File | Role | Paper Section |
|------|------|---------------|
| `env.py` | UAV Swarm topology, Lyapunov V_top, Gym-style step | §II-A, §IV |
| `model.py` | SplitFed MobileNetV3-small + Student KD model | §II-B |
| `agent.py` | SAC-CLF agent, CLF gatekeeper, replay buffer | §III–IV |
| `main.py` | Training loop, SplitFed gradient pipeline | §V (Alg 1) |

---

### Setup

```bash
pip install torch torchvision networkx numpy
# For Apple M2 MPS acceleration, PyTorch ≥ 2.0 is sufficient.
```

### Run

```bash
python main.py
```

Expected output per LOG_INTERVAL steps:
```
  Ep   1 | Step    0 | G-Step      0 |  R=-0.412  V_top=0.0231  η=0.500  L_cut= 7  SplitFed  T_loss=2.303  S_loss=2.198  CritL=nan  ActL=nan
```

---

### Key Mathematical Mapping

| Code | Equation |
|------|----------|
| `compute_v_top()` in `env.py` | Eq. (6) – Candidate Lyapunov function |
| `_update_eta()` in `env.py` | Eqs. (8)–(9) – Adaptive constraint strength |
| `check_lyapunov_constraint()` in `env.py` | Eq. (7) – CLF stability condition |
| `_utility_score()` in `env.py` | Eq. (11) – Rewiring utility score |
| `detect_dead_nodes()` in `env.py` | Eq. (12) – Adaptive timeout |
| `splitfed_cost()` in `model.py` | Eq. (13) – SplitFed cost function |
| `_compute_reward()` in `env.py` | Eq. (3)–(5) – Hybrid reward |
| `_clf_mask()` in `agent.py` | Algorithm 1, Steps 5–10 |
| `update()` → dampening term | Eq. (10) – Route-stability penalty |
| `run_splitfed_step()` in `main.py` | Algorithm 1, Steps 20–26 |

---

### Scaling Up

- Replace `mock_batch()` in `main.py` with a real `torch.utils.data.DataLoader` over EuroSAT.
- Increase `NUM_NODES`, `NUM_EPISODES`, `MAX_STEPS_EP`.
- Set `pretrained=True` in `SplitFedModel(...)` to start from ImageNet weights.
