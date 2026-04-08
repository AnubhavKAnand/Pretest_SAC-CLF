"""
env.py  –  UAV Swarm Topology Environment
==========================================
Implements the Physical Layer from Section II-A of the paper:
  "Robust Federated Deep Reinforcement Learning for Massive UAV Swarms
   via Hierarchical Small-World SplitFed Architecture"

Key responsibilities:
  - Maintain a dynamic Small-World graph G(t) = (V, E(t)) over N=20 nodes.
  - Expose per-drone state vectors  s_t  as defined in Eq. (1).
  - Provide a Gym-style step() that updates mock physics (battery, channel,
    queue length) and returns (next_state, reward, done, info).
  - Compute the candidate Lyapunov function V_top(e) from Eq. (6).
  - Implement Small-World rewiring (Eq. 11) and adaptive dead-node detection
    (Eq. 12) so the graph stays healthy across episodes.

Design note:
  Everything runs in a SINGLE PROCESS.  No sockets, no Ray, no multiprocessing.
  All "transmissions" between drones are simply Python function calls.
"""

import math
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Global constants matching the paper's notation
# ---------------------------------------------------------------------------
NUM_NODES        = 20          # N  – swarm size
K_NEIGHBORS      = 4           # k  – Watts-Strogatz initial ring neighbours
REWIRE_PROB      = 0.3         # p  – rewiring probability for Small-World
BASE_STATION_ID  = -1          # Sentinel: "Base Station" as offload target
P_TX_MAX         = 1.0         # P_max – normalised maximum transmit power
MAX_QUEUE        = 10          # max packets in local queue
BATTERY_FULL     = 100.0       # E_batt initial value (arbitrary units)
BATTERY_TX_BS    = 2.0         # energy cost to transmit to Base Station
BATTERY_TX_NEIGH = 0.5         # energy cost to transmit to a neighbour
PROC_TIME_BS     = 0.2         # T^BS_proc  – fast server compute (seconds)
PROC_TIME_NEIGH  = 0.8         # T^neigh_proc – slow neighbour compute
COMM_TIME_BASE   = 0.3         # base T_comm (scaled by channel quality later)
JAMMING_PROB     = 0.10        # probability a channel is jammed each step

# Lyapunov target topology metrics (Small-World "ideal")
C_TARGET         = 0.55        # target clustering coefficient
L_TARGET         = 2.5         # target average path length (hops)

# Adaptive η parameters (Eq. 9)
ETA_0            = 0.5         # base constraint decay rate
K_ETA            = 2.0         # sensitivity to network prediction error

# Utility score weights for rewiring (Eq. 11)
ALPHA_UTIL       = 0.4         # weight for Jaccard clustering term
BETA_UTIL        = 0.4         # weight for inverse hop-count term
GAMMA_UTIL       = 0.2         # weight for transmission energy penalty

# RTT-based dead-node detection (Eq. 12)
RTT_INIT_MEAN    = 0.3         # initial µ_RTT (seconds)
RTT_INIT_STD     = 0.05        # initial σ_RTT


# ---------------------------------------------------------------------------
# Helper: build initial Watts-Strogatz Small-World graph
# ---------------------------------------------------------------------------
def _build_small_world_graph(n: int, k: int, p: float, seed: int = 42) -> nx.Graph:
    """Return an undirected Watts-Strogatz graph with unit edge weights."""
    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    # Attach a default weight to every edge (used as channel-quality proxy)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


# ---------------------------------------------------------------------------
# Lyapunov Topology Metric  –  Eq. (6)
# ---------------------------------------------------------------------------
def compute_v_top(G: nx.Graph,
                  c_target: float = C_TARGET,
                  l_target: float = L_TARGET) -> Tuple[float, float, float]:
    """
    Compute the candidate Lyapunov function V_top(e) from Eq. (6):

        V_top(e) = 0.5*(C_meas - C_target)^2  +  0.5*(L_meas - L_target)^2

    Returns
    -------
    v_top   : scalar Lyapunov value
    c_meas  : measured average clustering coefficient
    l_meas  : measured average path length (over connected component)
    """
    c_meas = nx.average_clustering(G)

    # Path length is only defined on connected graphs; use largest component
    if nx.is_connected(G):
        l_meas = nx.average_shortest_path_length(G)
    else:
        giant = G.subgraph(max(nx.connected_components(G), key=len))
        l_meas = nx.average_shortest_path_length(giant)

    v_top = 0.5 * (c_meas - c_target) ** 2 + 0.5 * (l_meas - l_target) ** 2
    return v_top, c_meas, l_meas


# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------
class UAVSwarmEnv:
    """
    Single-process, time-stepped UAV swarm environment.

    State vector for drone i  (Eq. 1):
        s_t = { E_batt, Q_len, h_BS, h_neigh, σ_top }

    Action tuple for drone i  (Eq. 2):
        a_t = { P_tx ∈ [0, P_max],
                d_target ∈ {BS} ∪ N_neighbors,
                L_cut ∈ {1, …, L_max} }

    The environment is stepped once per (episode step, drone).  In main.py
    we loop over all drones each global step so every drone acts once per
    global timestep.
    """

    def __init__(self,
                 num_nodes: int = NUM_NODES,
                 max_episode_steps: int = 200,
                 seed: int = 42):
        self.num_nodes         = num_nodes
        self.max_episode_steps = max_episode_steps
        self.rng               = random.Random(seed)
        self.np_rng            = np.random.default_rng(seed)

        # Build initial topology
        self.G = _build_small_world_graph(num_nodes, K_NEIGHBORS, REWIRE_PROB, seed)

        # Per-drone physical state
        self.battery   = np.full(num_nodes, BATTERY_FULL)   # E_batt
        self.queue_len = np.zeros(num_nodes, dtype=int)      # Q_len
        # h_BS: channel quality to base station ∈ [0, 1]
        self.h_bs      = np.ones(num_nodes)
        # h_neigh[i]: dict mapping neighbour id → channel quality
        self.h_neigh: List[Dict[int, float]] = [
            {v: 1.0 for v in self.G.neighbors(i)} for i in range(num_nodes)
        ]

        # RTT statistics per link (for dead-node detection, Eq. 12)
        self.rtt_mean = np.full(num_nodes, RTT_INIT_MEAN)
        self.rtt_std  = np.full(num_nodes, RTT_INIT_STD)

        # Topology Lyapunov tracking
        self.v_top_prev, _, _ = compute_v_top(self.G)
        self.sigma_top        = self.v_top_prev   # included in state

        # Latency tracking for adaptive η (Eq. 8-9)
        self.latency_expected = COMM_TIME_BASE
        self.latency_actual   = COMM_TIME_BASE
        self.eta              = ETA_0             # adaptive constraint strength

        # Episode bookkeeping
        self.step_count = 0
        self.done       = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset environment; return initial state of drone 0."""
        self.__init__(self.num_nodes, self.max_episode_steps)
        return self._get_state(0)

    # ------------------------------------------------------------------
    # State construction  –  Eq. (1)
    # ------------------------------------------------------------------
    def _get_state(self, drone_id: int) -> np.ndarray:
        """
        Build the state vector for drone `drone_id`.

            s_t = [E_batt, Q_len, h_BS, *h_neigh_values, σ_top]

        h_neigh is padded/truncated to a fixed length of NUM_NODES so the
        vector has a constant dimension regardless of degree.
        """
        h_neigh_vals = list(self.h_neigh[drone_id].values())
        # Pad with zeros if fewer than NUM_NODES neighbours
        h_neigh_fixed = np.zeros(self.num_nodes)
        h_neigh_fixed[:len(h_neigh_vals)] = h_neigh_vals[:self.num_nodes]

        state = np.array([
            self.battery[drone_id] / BATTERY_FULL,   # normalised E_batt
            self.queue_len[drone_id] / MAX_QUEUE,     # normalised Q_len
            self.h_bs[drone_id],                      # h_BS
        ], dtype=np.float32)

        state = np.concatenate([state, h_neigh_fixed.astype(np.float32),
                                 [np.float32(self.sigma_top)]])
        return state   # shape: (3 + NUM_NODES + 1,) = (24,)

    @property
    def state_dim(self) -> int:
        return 3 + self.num_nodes + 1   # 24

    # ------------------------------------------------------------------
    # Channel & jamming simulation
    # ------------------------------------------------------------------
    def _refresh_channels(self):
        """
        Each timestep, channel qualities are re-sampled with optional jamming.
        Jamming zeroes a channel with probability JAMMING_PROB.
        """
        for i in range(self.num_nodes):
            if i not in self.G:
                continue
            # Channel to Base Station
            self.h_bs[i] = float(self.np_rng.uniform(0.3, 1.0))
            if self.np_rng.random() < JAMMING_PROB:
                self.h_bs[i] = 0.0          # jammed

            # Channels to neighbours
            for nb in list(self.G.neighbors(i)):
                q = float(self.np_rng.uniform(0.4, 1.0))
                if self.np_rng.random() < JAMMING_PROB:
                    q = 0.0
                self.h_neigh[i][nb] = q

    # ------------------------------------------------------------------
    # Latency & energy cost helpers  –  Eqs. (4), (5), (13)
    # ------------------------------------------------------------------
    def _transmission_time(self, drone_id: int, target: int,
                           data_size: float = 1.0) -> float:
        """
        Compute T_comm part of T_total.
        Uses Shannon capacity: T = DataSize / (B * log2(1 + SINR)).
        SINR is approximated by the channel quality h.
        B (bandwidth) is normalised to 1.
        """
        if target == BASE_STATION_ID:
            h = max(self.h_bs[drone_id], 1e-6)
        else:
            h = max(self.h_neigh[drone_id].get(target, 0.1), 1e-6)

        sinr    = h * 10.0          # simple linear SINR model
        rate    = math.log2(1 + sinr)
        t_comm  = data_size / rate
        return t_comm

    def _energy_cost(self, drone_id: int, target: int) -> float:
        """Return transmission energy E_tx based on destination type (Eq. 4)."""
        if target == BASE_STATION_ID:
            return BATTERY_TX_BS
        return BATTERY_TX_NEIGH

    def _proc_time(self, target: int) -> float:
        """Return processing time at destination (Eq. 5)."""
        if target == BASE_STATION_ID:
            return PROC_TIME_BS
        return PROC_TIME_NEIGH

    # ------------------------------------------------------------------
    # Reward  –  Eq. (3)
    # ------------------------------------------------------------------
    def _compute_reward(self, drone_id: int, target: int,
                        p_tx: float, w1: float = 0.5, w2: float = 0.5,
                        alpha: float = 0.1) -> float:
        """
        R(s_t, a_t) = -[w1*T_total + w2*E_total] + α*H(π)

        H(π) is approximated as a small entropy bonus proportional to how
        far p_tx is from the extremes (encourages exploration of power levels).
        """
        t_comm  = self._transmission_time(drone_id, target)
        t_proc  = self._proc_time(target)
        t_total = t_comm + t_proc

        e_total = self._energy_cost(drone_id, target) * p_tx

        # Entropy bonus H(π): maximised when p_tx ≈ 0.5
        h_pi = -abs(p_tx - 0.5) + 0.5   # ∈ [0, 0.5]

        reward = -(w1 * t_total + w2 * e_total) + alpha * h_pi
        return float(reward)

    # ------------------------------------------------------------------
    # Topology Lyapunov derivative  –  Eqs. (7)–(9)
    # ------------------------------------------------------------------
    def _update_eta(self):
        """
        Adaptive constraint strength η(t) from Eq. (9):
            δ_net(t)  = |Latency_expected − Latency_actual|
            η(t)      = η₀ * (1 + k_η * δ_net(t))
        """
        delta_net        = abs(self.latency_expected - self.latency_actual)
        self.eta         = ETA_0 * (1.0 + K_ETA * delta_net)

    def check_lyapunov_constraint(self, v_top_new: float,
                                  v_top_old: float) -> bool:
        """
        Check whether the CLF condition (Eq. 7) is satisfied:
            ΔV_top  ≤  -η(t) * V_top(old)

        Returns True if the constraint is SATISFIED (action is safe),
        False if the constraint is VIOLATED (action must be masked/blocked).
        """
        delta_v = v_top_new - v_top_old          # ΔV_top
        rhs     = -self.eta * v_top_old          # -η(t)*V_top
        return delta_v <= rhs

    # ------------------------------------------------------------------
    # Small-World rewiring  –  Eq. (11)
    # ------------------------------------------------------------------
    def _utility_score(self, drone_i: int, candidate_k: int) -> float:
        """
        Utility score U_k from Eq. (11):
            U_k = α * Jaccard(N_i, N_k)
                + β * 1/Hops(k, Server)
                - γ * E_tx(i, k)

        'Server' is approximated as the drone with the highest h_BS
        (closest proxy to the Base Station).
        """
        ni = set(self.G.neighbors(drone_i))
        nk = set(self.G.neighbors(candidate_k))
        union_   = ni | nk
        jaccard  = len(ni & nk) / max(len(union_), 1)

        # FIX: Find the "server proxy" ONLY among nodes currently alive in the graph
        alive_nodes = list(self.G.nodes())
        if not alive_nodes:
            hops = self.num_nodes
        else:
            server_proxy = max(alive_nodes, key=lambda n: self.h_bs[n])
            try:
                hops = nx.shortest_path_length(self.G, candidate_k, server_proxy)
            except nx.NetworkXNoPath:
                hops = self.num_nodes    # large penalty if no path

        e_tx = BATTERY_TX_NEIGH     # uniform cost approximation

        score = (ALPHA_UTIL * jaccard
                 + BETA_UTIL * (1.0 / max(hops, 1))
                 - GAMMA_UTIL * e_tx)
        return score

    def rewire_failed_node(self, failed_node: int):
        """
        When `failed_node` is detected as dead (Eq. 12), remove its edges
        and reconnect its former neighbours using the utility score (Eq. 11)
        subject to the Lyapunov acceptance criterion.
        """
        if failed_node not in self.G:
            return

        former_neighbours = list(self.G.neighbors(failed_node))
        self.G.remove_node(failed_node)

        # For each orphaned neighbour, find the best rewire candidate
        for nb in former_neighbours:
            if nb not in self.G:
                continue
            candidates = [n for n in self.G.nodes()
                          if n != nb and n not in self.G.neighbors(nb)]
            if not candidates:
                continue

            # Score & sort candidates
            scored = sorted(candidates,
                            key=lambda k: self._utility_score(nb, k),
                            reverse=True)

            v_before, _, _ = compute_v_top(self.G)
            for cand in scored[:5]:    # try top-5 to find one that passes CLF
                self.G.add_edge(nb, cand, weight=1.0)
                v_after, _, _ = compute_v_top(self.G)
                if self.check_lyapunov_constraint(v_after, v_before):
                    break              # accepted
                self.G.remove_edge(nb, cand)   # rejected, try next

    # ------------------------------------------------------------------
    # Dead-node detection  –  Eq. (12)
    # ------------------------------------------------------------------
    def detect_dead_nodes(self) -> List[int]:
        """
        A node i is considered 'dead' if its simulated RTT exceeds τ_timeout:
            τ_timeout(t) = µ_RTT(t) + 4 * σ_RTT(t)

        In this mock simulation we randomly spike a drone's RTT to trigger
        detection with low probability, simulating a crash event.
        """
        dead = []
        for i in range(self.num_nodes):
            if i not in self.G:
                continue
            # Simulate current RTT sample (normally distributed + occasional spike)
            rtt_sample = self.np_rng.normal(self.rtt_mean[i], self.rtt_std[i])
            if self.np_rng.random() < 0.02:        # 2% chance of large spike
                rtt_sample += self.np_rng.uniform(1.0, 3.0)

            tau = self.rtt_mean[i] + 4.0 * self.rtt_std[i]
            if rtt_sample > tau:
                dead.append(i)

            # Online RTT statistics update (exponential moving average)
            alpha_rtt       = 0.1
            self.rtt_mean[i] = (1 - alpha_rtt) * self.rtt_mean[i] + alpha_rtt * rtt_sample
            self.rtt_std[i]  = (1 - alpha_rtt) * self.rtt_std[i]  + alpha_rtt * abs(rtt_sample - self.rtt_mean[i])
        return dead

    # ------------------------------------------------------------------
    # Environment step  –  core Gym-style interface
    # ------------------------------------------------------------------
    def step(self, drone_id: int, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one action for `drone_id` and return (next_state, reward, done, info).

        Parameters
        ----------
        drone_id : int
            The acting drone.
        action   : dict with keys
            'p_tx'     – float ∈ [0, 1]      (normalised transmit power)
            'd_target' – int                  (BASE_STATION_ID or neighbour id)
            'l_cut'    – int ∈ [1, L_max]    (split layer index)

        Returns
        -------
        next_state : np.ndarray  shape (state_dim,)
        reward     : float
        done       : bool
        info       : dict  (diagnostic metrics)
        """
        assert not self.done, "Call reset() before stepping a finished episode."

        p_tx     = float(np.clip(action.get("p_tx", 0.5), 0.0, 1.0))
        d_target = action.get("d_target", BASE_STATION_ID)
        l_cut    = int(action.get("l_cut", 3))

        # ---- 1. Refresh channel qualities (simulates time-varying fading) ----
        self._refresh_channels()

        # ---- 2. Update adaptive η (Eq. 9) ------------------------------------
        # Actual latency is the computed transmission time
        self.latency_actual   = self._transmission_time(drone_id, d_target)
        self.latency_expected = COMM_TIME_BASE
        self._update_eta()

        # ---- 3. Lyapunov topology update -------------------------------------
        v_top_new, c_meas, l_meas = compute_v_top(self.G)
        self.sigma_top = v_top_new      # σ_top fed back into state

        # ---- 4. Battery drain ------------------------------------------------
        e_cost = self._energy_cost(drone_id, d_target) * p_tx
        self.battery[drone_id] = max(self.battery[drone_id] - e_cost, 0.0)

        # ---- 5. Queue update -------------------------------------------------
        # Each step we add 1 packet and attempt to drain via offloading
        self.queue_len[drone_id] = min(self.queue_len[drone_id] + 1, MAX_QUEUE)
        if d_target == BASE_STATION_ID or (drone_id in self.G and d_target in self.G.neighbors(drone_id)):
            self.queue_len[drone_id] = max(self.queue_len[drone_id] - 1, 0)

        # ---- 6. Reward -------------------------------------------------------
        reward = self._compute_reward(drone_id, d_target, p_tx)

        # ---- 7. Dead-node detection & rewiring --------------------------------
        dead_nodes = self.detect_dead_nodes()
        for dn in dead_nodes:
            self.rewire_failed_node(dn)

        # ---- 8. Episode termination ------------------------------------------
        self.step_count += 1
        
        # If a drone runs out of battery, it crashes (treat as a dead node). 
        # The rest of the swarm survives and rewires.
        if self.battery[drone_id] <= 0.0 and drone_id in self.G:
            self.rewire_failed_node(drone_id)
            
        # The episode only terminates early if the ENTIRE swarm dies
        if len(self.G.nodes) == 0:
            self.done = True

        next_state = self._get_state(drone_id)

        info = {
            "v_top"          : v_top_new,
            "c_meas"         : c_meas,
            "l_meas"         : l_meas,
            "eta"            : self.eta,
            "dead_nodes"     : dead_nodes,
            "latency_actual" : self.latency_actual,
            "battery"        : self.battery[drone_id],
        }
        return next_state, reward, self.done, info

    # ------------------------------------------------------------------
    # Reboot & Rejoin Logic
    # ------------------------------------------------------------------
    def _rejoin_swarm(self, node_id: int):
        """
        Attempts to securely re-integrate a recharged node back into the Small-World graph
        using the CLF utility score to find optimal neighbors.
        """
        self.G.add_node(node_id)
        self.rtt_mean[node_id] = RTT_INIT_MEAN  # Reset networking stats
        self.rtt_std[node_id]  = RTT_INIT_STD
        self.queue_len[node_id] = 0             # Clear dropped packets
        self.h_neigh[node_id]  = {}             # Clear stale channel data
        
        alive_nodes = [n for n in self.G.nodes() if n != node_id]
        if not alive_nodes:
            return  # Everyone else is dead, sit tight
            
        # Score all potential new neighbors
        candidates = sorted(alive_nodes, key=lambda k: self._utility_score(node_id, k), reverse=True)
        
        connected = 0
        v_before, _, _ = compute_v_top(self.G)
        
        # Try to establish K_NEIGHBORS connections, strictly gated by CLF
        for cand in candidates:
            self.G.add_edge(node_id, cand, weight=1.0)
            v_after, _, _ = compute_v_top(self.G)
            
            if self.check_lyapunov_constraint(v_after, v_before):
                v_before = v_after  # Accept and update baseline
                connected += 1
                if connected >= K_NEIGHBORS:
                    break
            else:
                self.G.remove_edge(node_id, cand)  # Blocked by CLF
                
        self.sigma_top = v_before

    def recharge_step(self, recharge_rate: float = 5.0):
        """
        Passively recharges dead nodes. If battery reaches FULL, the node reboots.
        """
        for i in range(self.num_nodes):
            if i not in self.G:
                self.battery[i] = min(self.battery[i] + recharge_rate, BATTERY_FULL)
                if self.battery[i] >= BATTERY_FULL:
                    self._rejoin_swarm(i)