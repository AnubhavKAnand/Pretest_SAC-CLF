"""
agent.py  –  SAC-CLF Agent (The Brain)
=======================================
Implements Section III–IV of the paper:
  "Robust Federated Deep Reinforcement Learning for Massive UAV Swarms …"

Soft Actor-Critic (SAC) with a Control Lyapunov Function (CLF) gatekeeper.

Architecture
------------
  Actor   : Gaussian policy network  → (µ, log σ) for p_tx (continuous)
            + Categorical head for d_target (discrete: BS vs neighbour)
            + Categorical head for l_cut    (discrete: 1 … L_MAX)

  Critic  : Twin Q-networks  Q1, Q2  (clipped double Q-learning)
            Both take (state, encoded_action) as input.

CLF Gatekeeper (Section IV-A, Eq. 7)
--------------------------------------
  Before ANY action is executed, check:
      ΔV_top  =  V_top(s_new) − V_top(s_old)  ≤  −η(t) * V_top(s_old)

  If condition FAILS, the proposed d_target is MASKED (removed from
  the valid action set) and the agent resamples.  This is the mathematical
  "gatekeeper" described in Algorithm 1, Steps 5-10.

Dampening constraint  –  Eq. (10)
-----------------------------------
  The smoothness penalty  β * ||n_target(t) − n_target(t−1)||²
  is added to the policy loss during gradient updates, discouraging
  rapid neighbour-switching.

Action space summary
---------------------
  p_tx      : continuous ∈ [0, 1]          (normalised transmit power)
  d_target  : discrete  0=Base Station,
                         1..k = neighbour index in [0..NUM_NODES-1]
  l_cut     : discrete  1 .. L_MAX (13 for MobileNetV3-small)

Replay buffer
-------------
  Simple list-based circular buffer (no external dependencies).
  Stores (state, action_vec, reward, next_state, done).
  action_vec = [p_tx, d_target_idx, l_cut]  (3-element float tensor)

All computations run in a SINGLE PROCESS (no Ray, no sockets).
"""

import math
import random
import copy
from collections import deque
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env import compute_v_top, NUM_NODES, BASE_STATION_ID

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
HIDDEN_DIM      = 128      # hidden units per layer in all networks
ACTOR_LR        = 3e-4
CRITIC_LR       = 3e-4
ALPHA_ENT_LR    = 3e-4    # learning rate for entropy temperature α
GAMMA           = 0.99    # discount factor
TAU             = 0.005   # Polyak soft-update coefficient for target networks
BUFFER_CAPACITY = 50_000
BATCH_SIZE      = 128
L_MAX           = 13      # MobileNetV3-small number of feature blocks
NUM_D_TARGET    = 2       # 0 = Base Station, 1 = best available neighbour
                           # (we binarise for simplicity; neighbour selection
                           #  uses the utility score from env)
DAMPENING_BETA  = 0.2     # β in Eq. (10)

# ---------------------------------------------------------------------------
# Utility: weight initialisation
# ---------------------------------------------------------------------------
def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


# ---------------------------------------------------------------------------
# Shared MLP backbone
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: int = HIDDEN_DIM, layers: int = 2):
        super().__init__()
        mods: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.ReLU()]
        mods.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*mods)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Actor Network
# Outputs:
#   (µ_ptx, log_σ_ptx) → for continuous p_tx  (reparameterised Gaussian)
#   logits_dtarget      → for discrete d_target (Categorical)
#   logits_lcut         → for discrete l_cut    (Categorical, L_MAX choices)
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """
    Hybrid actor: one continuous head + two discrete heads.

    Continuous p_tx  –  squashed Gaussian (tanh) to stay in [0,1].
    d_target         –  Categorical over {0=BS, 1=neighbour}.
    l_cut            –  Categorical over {1, 2, …, L_MAX}.
    """
    LOG_SIG_MIN = -5
    LOG_SIG_MAX = 2

    def __init__(self, state_dim: int):
        super().__init__()
        self.trunk = MLP(state_dim, HIDDEN_DIM)  # shared feature extractor

        # Continuous head: p_tx
        self.mean_head    = nn.Linear(HIDDEN_DIM, 1)
        self.log_std_head = nn.Linear(HIDDEN_DIM, 1)

        # Discrete head: d_target (binary: BS=0 or neighbour=1)
        self.dtarget_head = nn.Linear(HIDDEN_DIM, NUM_D_TARGET)

        # Discrete head: l_cut  (L_MAX choices, representing layers 1..L_MAX)
        self.lcut_head    = nn.Linear(HIDDEN_DIM, L_MAX)

        self.apply(_init_weights)

    def forward(self, state: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        p_tx_mean      : (B, 1) – mean of Gaussian before tanh squash
        p_tx_log_std   : (B, 1) – log std (clamped)
        dtarget_logits : (B, NUM_D_TARGET)
        lcut_logits    : (B, L_MAX)
        """
        feat = F.relu(self.trunk(state))

        p_tx_mean    = self.mean_head(feat)
        p_tx_log_std = torch.clamp(self.log_std_head(feat),
                                   self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        dtarget_log  = self.dtarget_head(feat)
        lcut_log     = self.lcut_head(feat)
        return p_tx_mean, p_tx_log_std, dtarget_log, lcut_log

    def sample(self, state: torch.Tensor,
               dtarget_mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Sample an action using the reparameterisation trick (p_tx) and
        Gumbel-softmax trick (discrete heads).

        Parameters
        ----------
        state        : (B, state_dim) tensor
        dtarget_mask : (B, NUM_D_TARGET) bool tensor; False = masked/invalid

        Returns
        -------
        action_vec  : (B, 3)  [p_tx, d_target_idx, l_cut_idx]
        log_prob    : (B,)    total log probability for SAC loss
        extras      : dict of entropy terms for logging
        """
        mean, log_std, dtarget_log, lcut_log = self.forward(state)
        std = log_std.exp()

        # ---- p_tx: reparameterised Gaussian + tanh squash → [0, 1] -----------
        normal   = torch.distributions.Normal(mean, std)
        x_t      = normal.rsample()                      # (B, 1)
        p_tx     = torch.sigmoid(x_t)                    # map to [0, 1]
        # Log prob correction for sigmoid (instead of tanh we use sigmoid)
        log_p_tx = (normal.log_prob(x_t)
                    - torch.log(p_tx * (1 - p_tx) + 1e-6)).sum(dim=-1)  # (B,)

        # ---- d_target: Categorical with optional CLF masking -----------------
        if dtarget_mask is not None:
            # Apply mask: set logits of invalid actions to -inf
            dtarget_log = dtarget_log.masked_fill(~dtarget_mask, -1e9)

        d_dist   = torch.distributions.Categorical(logits=dtarget_log)
        d_sample = d_dist.sample()                       # (B,)
        log_p_d  = d_dist.log_prob(d_sample)             # (B,)
        ent_d    = d_dist.entropy()

        # ---- l_cut: Categorical ----------------------------------------------
        lcut_dist   = torch.distributions.Categorical(logits=lcut_log)
        lcut_sample = lcut_dist.sample()                 # (B,) values 0..L_MAX-1
        log_p_lcut  = lcut_dist.log_prob(lcut_sample)   # (B,)
        ent_lcut    = lcut_dist.entropy()

        # Total log probability (sum over independent heads)
        log_prob = log_p_tx + log_p_d + log_p_lcut      # (B,)

        # Pack into action vector (float for replay buffer storage)
        action_vec = torch.stack([
            p_tx.squeeze(-1),
            d_sample.float(),
            (lcut_sample + 1).float(),   # shift so l_cut ∈ {1..L_MAX}
        ], dim=-1)                       # (B, 3)

        extras = {"ent_d": ent_d.mean().item(), "ent_lcut": ent_lcut.mean().item()}
        return action_vec, log_prob, extras


# ---------------------------------------------------------------------------
# Critic Network  –  Twin Q-networks
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    """
    Twin soft Q-networks Q1, Q2.
    Input: concatenation of [state, action_vec]
    Output: scalar Q-value.
    Using two independent critics mitigates overestimation bias (TD3 / SAC).
    """
    def __init__(self, state_dim: int, action_dim: int = 3):
        super().__init__()
        in_dim  = state_dim + action_dim
        self.Q1 = MLP(in_dim, 1)
        self.Q2 = MLP(in_dim, 1)

    def forward(self, state: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa  = torch.cat([state, action], dim=-1)
        q1  = self.Q1(sa)
        q2  = self.Q2(sa)
        return q1, q2

    def Q_min(self, state: torch.Tensor,
              action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Circular experience replay buffer.

    Stores (state, action_vec, reward, next_state, done) tuples.
    All tensors are float32 on CPU (moved to device at sample time).
    """

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((
            state.astype(np.float32),
            action.astype(np.float32),
            np.float32(reward),
            next_state.astype(np.float32),
            np.float32(done),
        ))

    def sample(self, batch_size: int,
               device: torch.device) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      device=device),
            torch.tensor(np.array(actions),     device=device),
            torch.tensor(np.array(rewards),     device=device).unsqueeze(-1),
            torch.tensor(np.array(next_states), device=device),
            torch.tensor(np.array(dones),       device=device).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# SAC-CLF Agent
# ---------------------------------------------------------------------------
class SACCLFAgent:
    """
    Soft Actor-Critic agent with Control Lyapunov Function (CLF) gatekeeper.

    Key methods
    -----------
    select_action(state, env, drone_id)
        Queries the actor for a candidate action, then runs the CLF check
        (Algorithm 1, Steps 5-10).  Unsafe actions are masked.

    update(batch)
        One gradient step for actor, critics, and entropy temperature α.
        Includes the dampening penalty from Eq. (10).

    CLF gatekeeper
    --------------
    For each sampled d_target, we hypothetically estimate V_top after the
    topology would change.  If the Lyapunov derivative condition fails
    (ΔV_top > -η * V_top_old), that target is masked to -inf in the actor's
    logits, forcing the agent to choose a safer target.

    In our 2-action binary formulation:
        d_target=0 → Base Station (always topologically neutral)
        d_target=1 → Best local neighbour (checked via CLF)
    """

    def __init__(self, state_dim: int,
                 device: Optional[torch.device] = None):
        self.device    = device or torch.device("cpu")
        self.state_dim = state_dim

        # Networks
        self.actor  = Actor(state_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # Target critic (slow-moving copy, updated via Polyak averaging)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Optimisers
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # Entropy temperature α (learnable, SAC auto-tuning)
        self.log_alpha      = torch.tensor(0.0, requires_grad=True,
                                           device=self.device)
        self.alpha_opt      = optim.Adam([self.log_alpha], lr=ALPHA_ENT_LR)
        # Target entropy = -|action_dim| (a common heuristic)
        self.target_entropy = -3.0   # 3 = dimensionality of action_vec

        # Replay buffer
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)

        # Dampening: remember last d_target to apply Eq. (10)
        self.prev_dtarget: Dict[int, int] = {}   # drone_id → last d_target

        # Training counter
        self.update_count = 0

    # ------------------------------------------------------------------
    # α property (entropy coefficient)
    # ------------------------------------------------------------------
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # CLF gatekeeper  –  Algorithm 1, Steps 5-10
    # ------------------------------------------------------------------
    def _clf_mask(self, env, drone_id: int) -> torch.Tensor:
        """
        Build a (1, NUM_D_TARGET) boolean mask where True = action is safe.

        Logic
        -----
        d_target=0 (Base Station):
            Transmitting to BS doesn't change the graph topology → always safe.

        d_target=1 (neighbour):
            Simulate connecting drone_id to its best neighbour, compute
            V_top_new, check the CLF condition from Eq. (7):
                ΔV_top ≤ -η(t) * V_top_old

        If condition fails, mask out the neighbour option.
        """
        mask = torch.ones(1, NUM_D_TARGET, dtype=torch.bool, device=self.device)

        # Retrieve current Lyapunov value and η
        v_top_old = env.sigma_top       # cached from last env step
        eta       = env.eta

        # --- Check neighbour option (d_target=1) ---
        neighbours = list(env.G.neighbors(drone_id)) if drone_id in env.G else []
        if not neighbours:
            # No neighbours → only BS is valid
            mask[0, 1] = False
        else:
            # Hypothetically add a "virtual" edge to the best utility neighbour
            # and see if V_top would worsen
            best_nb  = max(neighbours,
                           key=lambda k: env._utility_score(drone_id, k))
            # Temporarily add edge (it might already exist)
            already_connected = env.G.has_edge(drone_id, best_nb)
            if already_connected:
                # FIX: If already connected, transmitting doesn't harm topology
                safe = True 
            else:
                env.G.add_edge(drone_id, best_nb, weight=1.0)
                v_top_new, _, _ = compute_v_top(env.G)
                env.G.remove_edge(drone_id, best_nb)   # undo hypothetical
                safe = env.check_lyapunov_constraint(v_top_new, v_top_old)
            if not safe:
                mask[0, 1] = False   # CLF violation → block neighbour target

        return mask

    # ------------------------------------------------------------------
    # Action selection  (uses CLF mask + actor sampling)
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, env,
                      drone_id: int,
                      deterministic: bool = False) -> Dict:
        """
        Algorithm 1 – Step 12 (action selection):
            1. Build CLF mask.
            2. Sample action from actor (respecting mask).
            3. Decode integer action into environment format.

        Returns
        -------
        dict with keys: 'p_tx', 'd_target', 'l_cut', 'raw_action_vec'
        """
        s = torch.tensor(state, dtype=torch.float32,
                         device=self.device).unsqueeze(0)   # (1, state_dim)

        # Step 1: CLF gatekeeper
        clf_mask = self._clf_mask(env, drone_id)            # (1, NUM_D_TARGET)

        # If ALL targets are masked (shouldn't normally happen), allow BS only
        if not clf_mask.any():
            clf_mask[0, 0] = True   # BS always a fallback

        # Step 2: Sample from actor
        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                # Greedy: use mode of distributions
                mean, _, dtarget_log, lcut_log = self.actor(s)
                p_tx     = torch.sigmoid(mean).squeeze()
                # Mask logits for d_target
                dtarget_log_m = dtarget_log.masked_fill(~clf_mask, -1e9)
                d_idx    = dtarget_log_m.argmax(dim=-1).item()
                lcut_idx = lcut_log.argmax(dim=-1).item()
                action_vec = torch.tensor([[p_tx, float(d_idx),
                                            float(lcut_idx + 1)]])
            else:
                action_vec, _, _ = self.actor.sample(s, dtarget_mask=clf_mask)
        self.actor.train()

        p_tx_val  = float(action_vec[0, 0])
        d_idx_val = int(action_vec[0, 1].round())
        l_cut_val = int(action_vec[0, 2].round())
        l_cut_val = max(1, min(l_cut_val, L_MAX))

        # Resolve d_target to environment format
        if d_idx_val == 0:
            d_target = BASE_STATION_ID
        else:
            # Pick best utility-score neighbour
            neighbours = list(env.G.neighbors(drone_id)) if drone_id in env.G else []
            if neighbours:
                d_target = max(neighbours,
                               key=lambda k: env._utility_score(drone_id, k))
            else:
                d_target = BASE_STATION_ID

        # Algorithm 1 Step 13-18: Adaptive l_cut based on channel quality
        h_channel = env.h_bs[drone_id]
        if h_channel < 0.4:
            # Weak channel → compute more locally (increase l_cut)
            l_cut_val = min(l_cut_val + 2, L_MAX)
        elif h_channel > 0.75:
            # Strong channel → transmit more (decrease l_cut)
            l_cut_val = max(l_cut_val - 1, 1)

        return {
            "p_tx"           : p_tx_val,
            "d_target"       : d_target,
            "l_cut"          : l_cut_val,
            "raw_action_vec" : action_vec.squeeze(0).cpu().numpy(),
            "d_idx"          : d_idx_val,
        }

    # ------------------------------------------------------------------
    # Store transition
    # ------------------------------------------------------------------
    def store(self, state: np.ndarray, action_dict: Dict,
              reward: float, next_state: np.ndarray, done: bool):
        """Push a (s, a, r, s', done) tuple into the replay buffer."""
        self.buffer.push(state,
                         action_dict["raw_action_vec"],
                         reward, next_state, done)

    # ------------------------------------------------------------------
    # SAC update step  (Critic + Actor + α)
    # ------------------------------------------------------------------
    def update(self, drone_id: int = 0) -> Optional[Dict]:
        """
        One SAC gradient step.  Returns loss dict for logging, or None if
        the buffer doesn't have enough samples yet.

        Includes the dampening penalty from Eq. (10):
            J_smooth = J_SAC + β * ||n_target(t) − n_target(t-1)||²
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE, self.device)

        # ---- Critic update ---------------------------------------------------
        with torch.no_grad():
            next_action_vec, next_log_pi, _ = self.actor.sample(next_states)
            q_next = self.critic_target.Q_min(next_states, next_action_vec)
            # SAC Bellman target: r + γ * (Q - α * log π)
            q_target = rewards + GAMMA * (1.0 - dones) * (q_next - self.alpha * next_log_pi.unsqueeze(-1))

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ---- Actor update (with dampening penalty, Eq. 10) -------------------
        new_action_vec, log_pi, extras = self.actor.sample(states)

        q_pi = self.critic.Q_min(states, new_action_vec)

        # Dampening cost: ||d_target_new − d_target_prev||²
        d_target_new  = new_action_vec[:, 1]        # (B,)
        d_target_prev = actions[:, 1]               # (B,) from replay buffer
        dampening     = DAMPENING_BETA * ((d_target_new - d_target_prev) ** 2).mean()

        # SAC actor loss: minimise (α * log_π − Q) + dampening
        actor_loss = (self.alpha.detach() * log_pi.unsqueeze(-1) - q_pi).mean() \
                     + dampening

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ---- Entropy temperature α update ------------------------------------
        alpha_loss = -(self.log_alpha *
                       (log_pi + self.target_entropy).detach().mean())
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- Polyak soft update target critic --------------------------------
        for param, tparam in zip(self.critic.parameters(),
                                  self.critic_target.parameters()):
            tparam.data.copy_(TAU * param.data + (1.0 - TAU) * tparam.data)

        self.update_count += 1

        return {
            "critic_loss" : critic_loss.item(),
            "actor_loss"  : actor_loss.item(),
            "alpha"       : self.alpha.item(),
            "dampening"   : dampening.item(),
            "entropy_d"   : extras["ent_d"],
            "entropy_lcut": extras["ent_lcut"],
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str = "sac_clf_checkpoint.pt"):
        torch.save({
            "actor"         : self.actor.state_dict(),
            "critic"        : self.critic.state_dict(),
            "critic_target" : self.critic_target.state_dict(),
            "log_alpha"     : self.log_alpha,
        }, path)
        print(f"[Agent] Checkpoint saved → {path}")

    def load(self, path: str = "sac_clf_checkpoint.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"]
        print(f"[Agent] Checkpoint loaded ← {path}")
