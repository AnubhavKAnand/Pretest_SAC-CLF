"""
main.py  –  HSW-SF Training Loop (Single-Process)
==================================================
Ties together:
    env.py   → UAVSwarmEnv      (Small-World topology + Lyapunov metrics)
    model.py → SplitFedModel    (MobileNetV3-small with dynamic split)
               StudentModel     (KD fallback)
    agent.py → SACCLFAgent      (SAC + CLF gatekeeper)

Execution model
---------------
SINGLE PROCESS, TIME-STEPPED.  No multiprocessing, no sockets, no Ray.

Each "global step" represents one discrete time slot t in which:
  1. Every drone i=0..N-1 queries the SAC-CLF agent for an action.
  2. The CLF gatekeeper screens unsafe topology changes.
  3. The environment is stepped; reward and next-state are stored.
  4. A mock "SplitFed gradient pipeline" is run:
       - One representative drone (drone_id=0) acts as the "active learner".
       - client_forward() simulates the drone computing the smashed data.
       - server_forward() simulates the base station completing the forward pass.
       - A cross-entropy loss is computed and backprop runs naturally through
         both halves (autograd handles this because everything is in-process).
       - The student model is updated via KD using the teacher's soft labels.
  5. Agent.update() performs one SAC gradient step.
  6. Metrics are logged to console every LOG_INTERVAL steps.

Key design choices
------------------
• Mock image data:  We generate random (B, 3, 64, 64) tensors to avoid
  needing the EuroSAT dataset.  Replace `mock_batch()` with a real
  DataLoader for a full experiment.
• Drone 0 is the "active SplitFed learner" each step; in a real deployment,
  all drones would take turns, but for a single-machine pretest this is
  sufficient to exercise every code path.
• Model optimisers are kept separate from the RL agent's optimisers.
"""

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os

from env   import UAVSwarmEnv, BASE_STATION_ID, NUM_NODES
from model import SplitFedModel, StudentModel, NUM_CLASSES, INPUT_SHAPE
from agent import SACCLFAgent
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT
from torch.utils.data import DataLoader
# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
SEED              = 42
NUM_EPISODES      = 30        # total training episodes  (increase for real runs)
MAX_STEPS_EP      = 100       # max environment steps per episode
WARMUP_STEPS      = 512       # fill buffer before training begins
ACTIVE_DRONE      = 0         # which drone runs the SplitFed model gradient step
BATCH_SZ_IMG      = 4         # mini-batch size for SplitFed gradient step
MODEL_LR          = 1e-3      # learning rate for SplitFed teacher model
STUDENT_LR        = 5e-4      # learning rate for Student (KD) model
LOG_INTERVAL      = 10        # print metrics every N global steps
SAVE_INTERVAL     = 5         # save agent checkpoint every N episodes
KD_TEMPERATURE    = 4.0       # temperature for Knowledge Distillation
KD_ALPHA          = 0.7       # KD loss mixing coefficient
DEVICE            = torch.device("mps" if torch.backends.mps.is_available()
                                 else "cpu")   # MPS = Apple M2 GPU

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Mock image batch generator (replaces a real DataLoader)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Real EuroSAT DataLoader
# ---------------------------------------------------------------------------
def get_eurosat_dataloader(batch_size: int = BATCH_SZ_IMG):
    """
    Downloads and loads the EuroSAT dataset.
    Applies standard ImageNet normalisation since we are using MobileNet.
    """
    transform = transforms.Compose([
        transforms.Resize(INPUT_SHAPE[1:]),  # Resize to 64x64
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Download=True will fetch it automatically the first time you run it
    print("[Data] Loading EuroSAT dataset (this may take a moment if downloading...)")
    dataset = EuroSAT(root="./data", download=True, transform=transform)
    
    # drop_last=True ensures we always get a consistent batch size
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

# ---------------------------------------------------------------------------
# SplitFed gradient pipeline
# ---------------------------------------------------------------------------
def run_splitfed_step(teacher: SplitFedModel,
                      student: StudentModel,
                      teacher_opt: optim.Optimizer,
                      student_opt: optim.Optimizer,
                      images: torch.Tensor,       # <--- NEW ARGUMENT
                      labels: torch.Tensor,       # <--- NEW ARGUMENT
                      l_cut: int,
                      use_student_fallback: bool = False) -> dict:
    """
    Simulates one mini-batch forward-backward pass through the SplitFed pipeline.

    Multi-hop simulation (Section V Algorithm 1, Steps 20-26):
    ----------------------------------------------------------
    Normal mode (Algorithm 1 Step 20-23):
        1. [Drone]  client_forward(x, l_cut)  → smashed_data
        2. [Server] server_forward(smashed_data, l_cut) → logits
        3. [Server] compute CE loss, backprop through server layers.
        4. [Drone]  gradient flows back through client layers via autograd.
        5. [Server] sends "soft labels" to drone → drone updates Student model.

    Fallback mode (Algorithm 1 Step 24-26):
        If `use_student_fallback=True` (CLF detected instability):
            Only the Student model runs locally.  No transmission cost.

    Returns
    -------
    dict with teacher_loss, student_loss, l_cut_used
    """
    #images, labels = mock_batch()
    ce_loss_fn = nn.CrossEntropyLoss()

    teacher_loss_val = 0.0
    student_loss_val = 0.0

    if not use_student_fallback:
        # ---- Phase 1: Client side (drone) ------------------------------------
        teacher_opt.zero_grad()
        smashed = teacher.client_forward(images, l_cut)
        # smashed.retain_grad()   # optionally inspect gradients at cut layer

        # ---- Phase 2: Server side (BS or neighbour) --------------------------
        logits = teacher.server_forward(smashed, l_cut)

        # ---- Backward pass (autograd connects both halves naturally) ---------
        teacher_loss = ce_loss_fn(logits, labels)
        teacher_loss.backward()
        teacher_opt.step()
        teacher_loss_val = teacher_loss.item()

        # ---- Knowledge Distillation: update Student model --------------------
        # Soft labels from teacher (Step 22: "Server sends Soft Labels")
        soft = teacher.soft_labels(images, l_cut=l_cut, temperature=KD_TEMPERATURE)

        student_opt.zero_grad()
        s_logits     = student(images)
        student_loss = StudentModel.kd_loss(s_logits, soft, labels,
                                            temperature=KD_TEMPERATURE,
                                            alpha=KD_ALPHA)
        student_loss.backward()
        student_opt.step()
        student_loss_val = student_loss.item()

    else:
        # ---- Fallback: Student-only (Step 24-26) ----------------------------
        student_opt.zero_grad()
        s_logits     = student(images)
        student_loss = ce_loss_fn(s_logits, labels)
        student_loss.backward()
        student_opt.step()
        student_loss_val = student_loss.item()

    return {
        "teacher_loss" : teacher_loss_val,
        "student_loss" : student_loss_val,
        "l_cut"        : l_cut,
        "fallback"     : use_student_fallback,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train():
    print("=" * 65)
    print("  HSW-SF Pretest Training Loop")
    print(f"  Device : {DEVICE}")
    print(f"  Nodes  : {NUM_NODES}  |  Episodes : {NUM_EPISODES}"
          f"  |  Steps/ep : {MAX_STEPS_EP}")
    print("=" * 65)

    # ---- Initialise components ----------------------------------------------
    env     = UAVSwarmEnv(num_nodes=NUM_NODES,
                          max_episode_steps=MAX_STEPS_EP,
                          seed=SEED)

    teacher = SplitFedModel(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    student = StudentModel(num_classes=NUM_CLASSES).to(DEVICE)
    teacher.train()
    student.train()

    teacher_opt = optim.Adam(teacher.parameters(), lr=MODEL_LR)
    student_opt = optim.Adam(student.parameters(), lr=STUDENT_LR)

    agent = SACCLFAgent(state_dim=env.state_dim, device=DEVICE)

    dataloader = get_eurosat_dataloader()
    data_iter = iter(dataloader)

    # ---- Metric accumulators ------------------------------------------------
    global_step    = 0
    episode_returns: list = []

    # ---- Training -----------------------------------------------------------
    for ep in range(1, NUM_EPISODES + 1):
        state  = env.reset()         # initial state of drone 0
        ep_ret = 0.0
        ep_t0  = time.time()

        # Collect per-episode metrics
        ep_vtop:  list = []
        ep_eta:   list = []
        ep_crit:  list = []
        ep_act:   list = []
        ep_split: list = []

        for step in range(MAX_STEPS_EP):

            # -- Step all drones (round-robin in single process) ---------------
            # Only drone ACTIVE_DRONE's state/return is tracked here for brevity;
            # other drones still step the environment so topology evolves.
            all_actions = {}
            for drone_id in range(NUM_NODES):
                if drone_id not in env.G:
                    continue   # dead node; skip

                # Get state for this drone
                s_i = env._get_state(drone_id)

                # Warm-up: random actions before training starts
                if global_step < WARMUP_STEPS:
                    neighbours = list(env.G.neighbors(drone_id))
                    d_tgt = BASE_STATION_ID if (not neighbours or random.random() < 0.5) \
                            else random.choice(neighbours)
                    action_i = {
                        "p_tx"           : random.random(),
                        "d_target"       : d_tgt,
                        "l_cut"          : random.randint(1, 13),
                        "raw_action_vec" : np.array([random.random(),
                                                     0.0 if d_tgt == BASE_STATION_ID else 1.0,
                                                     float(random.randint(1, 13))]),
                        "d_idx"          : 0 if d_tgt == BASE_STATION_ID else 1,
                    }
                else:
                    action_i = agent.select_action(s_i, env, drone_id)

                all_actions[drone_id] = (s_i, action_i)

            # -- Step environment for each drone -------------------------------
            # We use drone ACTIVE_DRONE's transition for RL update & metrics.
            main_next_state = state
            main_reward     = 0.0
            main_done       = False
            main_info       = {}

            for drone_id, (s_i, action_i) in all_actions.items():
                env_action = {
                    "p_tx"     : action_i["p_tx"],
                    "d_target" : action_i["d_target"],
                    "l_cut"    : action_i["l_cut"],
                }
                next_s, rew, done, info = env.step(drone_id, env_action)

                # Store transition for active drone
                agent.store(s_i, action_i, rew, next_s, done)

                if drone_id == ACTIVE_DRONE:
                    main_next_state = next_s
                    main_reward     = rew
                    main_done       = done
                    main_info       = info
                    ep_vtop.append(info["v_top"])
                    ep_eta.append(info["eta"])
            
            env.recharge_step(recharge_rate=1.5)

            ep_ret += main_reward

            # -- SplitFed gradient step (Algorithm 1 Steps 20-26) -------------
            # Determine if we should use student fallback:
            #   Fallback triggers when the CLF detects a topology violation.
            #   We use sigma_top as a proxy: if it's too high, we're unstable.
            use_fallback = (env.sigma_top > 0.5)

            active_action = all_actions.get(ACTIVE_DRONE)
            l_cut_used    = active_action[1]["l_cut"] if active_action else 7

            try:
                images, labels = next(data_iter)
            except StopIteration:
                # If we run out of images in the epoch, restart the iterator
                data_iter = iter(dataloader)
                images, labels = next(data_iter)

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            split_metrics = run_splitfed_step(
                teacher, student, teacher_opt, student_opt,
                images, labels,  # <--- PASSING DATA HERE
                l_cut=l_cut_used, use_student_fallback=use_fallback
            )
            ep_split.append(split_metrics)

            # -- SAC update ----------------------------------------------------
            if global_step >= WARMUP_STEPS:
                rl_metrics = agent.update(drone_id=ACTIVE_DRONE)
                if rl_metrics:
                    ep_crit.append(rl_metrics["critic_loss"])
                    ep_act.append(rl_metrics["actor_loss"])

            # -- Logging -------------------------------------------------------
            if global_step % LOG_INTERVAL == 0:
                avg_crit  = np.mean(ep_crit)  if ep_crit  else float("nan")
                avg_act   = np.mean(ep_act)   if ep_act   else float("nan")
                avg_vtop  = np.mean(ep_vtop)  if ep_vtop  else float("nan")
                t_loss    = split_metrics["teacher_loss"]
                s_loss    = split_metrics["student_loss"]
                print(
                    f"  Ep {ep:3d} | Step {step:4d} | G-Step {global_step:6d} |"
                    f"  R={main_reward:+.3f}"
                    f"  V_top={avg_vtop:.4f}"
                    f"  η={env.eta:.3f}"
                    f"  L_cut={l_cut_used:2d}"
                    f"  {'FALLBACK' if use_fallback else 'SplitFed'}"
                    f"  T_loss={t_loss:.3f}"
                    f"  S_loss={s_loss:.3f}"
                    f"  CritL={avg_crit:.4f}"
                    f"  ActL={avg_act:.4f}"
                )

            global_step += 1
            state = main_next_state

            if main_done:
                break
        # ---- End of episode --------------------------------------------------

        # Calculate episode averages for the paper graphs
        avg_vtop = np.mean(ep_vtop) if ep_vtop else 0.0
        max_eta  = np.max(ep_eta) if ep_eta else 0.0
        
        avg_t_loss = np.mean([m["teacher_loss"] for m in ep_split]) if ep_split else 0.0
        avg_s_loss = np.mean([m["student_loss"] for m in ep_split]) if ep_split else 0.0
        
        # Calculate physical metrics
        avg_battery_left = np.mean(env.battery)  # Higher is better
        avg_queue_len = np.mean(env.queue_len)   # Lower is better (Latency)

        # Write to CSV
        csv_file = "btp_results.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Episode", "Return", "Avg_V_top", "Max_Eta", "T_Loss", "S_Loss", "Avg_Battery", "Avg_Queue"])
            writer.writerow([ep, ep_ret, avg_vtop, max_eta, avg_t_loss, avg_s_loss, avg_battery_left, avg_queue_len])
        
        episode_returns.append(ep_ret)
        elapsed = time.time() - ep_t0
        avg_100 = np.mean(episode_returns[-min(10, len(episode_returns)):])
        print(
            f"\n>>> Episode {ep:3d} done  |"
            f"  Return={ep_ret:.2f}"
            f"  Avg10={avg_100:.2f}"
            f"  Time={elapsed:.1f}s"
            f"  Buffer={len(agent.buffer)}"
            f"  Alive_nodes={env.G.number_of_nodes()}/{NUM_NODES}\n"
        )

        # Save checkpoint
        if ep % SAVE_INTERVAL == 0:
            agent.save(f"sac_clf_ep{ep:03d}.pt")

    # ---- Training complete ---------------------------------------------------
    print("\n" + "=" * 65)
    print("  Training complete.")
    print(f"  Final avg return (last 10 episodes): "
          f"{np.mean(episode_returns[-10:]):.3f}")
    print("=" * 65)

    return episode_returns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    returns = train()
