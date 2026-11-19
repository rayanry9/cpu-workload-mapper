import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PointerActorCritic(nn.Module):
    def __init__(self, core_dim=6, global_dim=6, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model

        # 1. Embed each core [x, y, area, is_p_core, assigned_power, is_occupied]
        self.core_embed = nn.Sequential(
            nn.Linear(core_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 2. Self-attention across cores (learns spatial/thermal neighbor interactions)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 3. Pointer head: query from current workload + queue, keys from encoded cores
        self.query_proj = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

        # 4. Critic head: mean-pools core embeddings + query -> scalar state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, core_feats, global_feats, action_mask, pad_mask=None):
        """
        core_feats:   (B, N, 6)  per-core features
        global_feats: (B, 6)     [cur_e, cur_p, rem_mean_p, rem_max_p, free_e_frac, free_p_frac]
        action_mask:  (B, N)     bool, True = valid free core, False = occupied/padded
        pad_mask:     (B, N)     optional bool, True = padded dummy core (when batching mixed N)
        """
        h = self.core_embed(core_feats)  # (B, N, d_model)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        # Pointer logits: (B, 1, d) @ (B, d, N) -> (B, N)
        q = self.query_proj(global_feats).unsqueeze(1)
        k = self.key_proj(h)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)

        # Mask occupied (and padded) cores to -inf before Categorical softmax
        logits = logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=logits)

        # Mean-pool only real cores for the value estimate
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(-1).float()
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)

        value = self.critic(torch.cat([pooled, q.squeeze(1)], dim=-1)).squeeze(-1)
        return dist, value


def pad_step(core_feats, action_mask, n_max):
    """Pad (N, 6) cores and (N,) mask up to n_max for batching across mixed floorplans."""
    n = core_feats.shape[0]
    pad_len = n_max - n
    pad_mask = np.zeros(n_max, dtype=bool)
    if pad_len > 0:
        core_feats = np.pad(core_feats, ((0, pad_len), (0, 0)), mode="constant")
        action_mask = np.pad(
            action_mask, (0, pad_len), mode="constant", constant_values=False
        )
        pad_mask[n:] = True
    return core_feats, action_mask, pad_mask


def compute_gae(rewards, values, dones, next_value, gamma=1.0, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation (GAE) and returns."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - float(dones[t])
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def train_masked_ppo(
    env,
    policy,
    n_max,
    total_updates=200,
    steps_per_rollout=512,
    epochs=4,
    batch_size=64,
    lr=3e-4,
    gamma=1.0,  # 1.0 works well when reward arrives only at the end of N steps
    gae_lambda=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    device="cpu",
):
    policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
    best_t_max = float("inf")

    obs, _ = env.reset()  # Expects obs = (core_feats (N,6), global_feats (6,))

    for update in range(1, total_updates + 1):
        # --- 1. Rollout collection ---
        buf_cores, buf_globals, buf_masks, buf_pads = [], [], [], []
        buf_actions, buf_logprobs, buf_rewards, buf_dones, buf_values = (
            [],
            [],
            [],
            [],
            [],
        )
        ep_temps = []

        for _ in range(steps_per_rollout):
            core_feats, global_feats = obs
            mask = env.action_masks()
            c_pad, m_pad, p_pad = pad_step(core_feats, mask, n_max)

            c_t = torch.tensor(c_pad, dtype=torch.float32, device=device).unsqueeze(0)
            g_t = torch.tensor(
                global_feats, dtype=torch.float32, device=device
            ).unsqueeze(0)
            m_t = torch.tensor(m_pad, dtype=torch.bool, device=device).unsqueeze(0)
            p_t = torch.tensor(p_pad, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                dist, val = policy(c_t, g_t, m_t, pad_mask=p_t)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done, _, info = env.step(int(action.item()))

            buf_cores.append(c_pad)
            buf_globals.append(global_feats)
            buf_masks.append(m_pad)
            buf_pads.append(p_pad)
            buf_actions.append(int(action.item()))
            buf_logprobs.append(float(log_prob.item()))
            buf_rewards.append(float(reward))
            buf_dones.append(bool(done))
            buf_values.append(float(val.item()))

            if done:
                if "t_max" in info:
                    ep_temps.append(info["t_max"])
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap value for the state right after the rollout ends
        with torch.no_grad():
            if buf_dones[-1]:
                next_value = 0.0
            else:
                c_pad, m_pad, p_pad = pad_step(obs[0], env.action_masks(), n_max)
                _, nv = policy(
                    torch.tensor(c_pad, dtype=torch.float32, device=device).unsqueeze(
                        0
                    ),
                    torch.tensor(obs[1], dtype=torch.float32, device=device).unsqueeze(
                        0
                    ),
                    torch.tensor(m_pad, dtype=torch.bool, device=device).unsqueeze(0),
                    pad_mask=torch.tensor(
                        p_pad, dtype=torch.bool, device=device
                    ).unsqueeze(0),
                )
                next_value = float(nv.item())

        # --- 2. GAE & returns ---
        adv, ret = compute_gae(
            np.array(buf_rewards, dtype=np.float32),
            np.array(buf_values, dtype=np.float32),
            np.array(buf_dones, dtype=bool),
            next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Convert buffers to tensors
        t_cores = torch.tensor(np.array(buf_cores), dtype=torch.float32, device=device)
        t_globals = torch.tensor(
            np.array(buf_globals), dtype=torch.float32, device=device
        )
        t_masks = torch.tensor(np.array(buf_masks), dtype=torch.bool, device=device)
        t_pads = torch.tensor(np.array(buf_pads), dtype=torch.bool, device=device)
        t_actions = torch.tensor(buf_actions, dtype=torch.long, device=device)
        t_old_logprobs = torch.tensor(buf_logprobs, dtype=torch.float32, device=device)
        t_adv = torch.tensor(adv, dtype=torch.float32, device=device)
        t_ret = torch.tensor(ret, dtype=torch.float32, device=device)

        # --- 3. Clipped PPO update ---
        idxs = np.arange(steps_per_rollout)
        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, steps_per_rollout, batch_size):
                mb = idxs[start : start + batch_size]
                dist, new_val = policy(
                    t_cores[mb], t_globals[mb], t_masks[mb], pad_mask=t_pads[mb]
                )
                new_logprob = dist.log_prob(t_actions[mb])
                ratio = torch.exp(new_logprob - t_old_logprobs[mb])

                surr1 = ratio * t_adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * t_adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((new_val - t_ret[mb]) ** 2).mean()
                entropy_loss = -dist.entropy().mean()

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if ep_temps:
            mean_t = float(np.mean(ep_temps))
            if mean_t < best_t_max:
                best_t_max = mean_t
                torch.save(policy.state_dict(), "best_hotspot_mapper.pt")
        if update % 10 == 0 and ep_temps:
            print(
                f"Update {update:3d} | Mean T_max: {np.mean(ep_temps):.2f} | Min T_max: {np.min(ep_temps):.2f}"
            )


def parse_flp(
    flp_path,
    p_prefixes=("pcore", "p_core", "big", "core_0_", "core_1_"),
    e_prefixes=("ecore", "e_core", "little", "core_"),
    default_fixed_power=None,
):
    """
    Parse a HotSpot 7.0 .flp file into normalized core features and fixed blocks.

    Returns dict with:
      - core_names: list[str] of length N
      - core_xy_area: np.ndarray of shape (N, 3) -> [norm_cx, norm_cy, norm_area] in [0, 1]
      - core_types: np.ndarray of shape (N,) -> 1 = P-core, 0 = E-core
      - fixed_blocks: dict[str, float] -> non-core block names mapped to background power (W)
      - all_blocks: list[str] -> all block names in .flp order
    """
    default_fixed_power = default_fixed_power or {}
    blocks = []

    with open(flp_path, "r") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            name = parts[0]
            w, h, x, y = map(float, parts[1:5])
            blocks.append((name, w, h, x, y))

    if not blocks:
        raise ValueError(f"No valid blocks found in {flp_path}")

    # Chip bounding box for [0, 1] coordinate normalization
    min_x = min(x for _, _, _, x, _ in blocks)
    min_y = min(y for _, _, _, _, y in blocks)
    max_x = max(x + w for _, w, _, x, _ in blocks)
    max_y = max(y + h for _, _, h, x, y in blocks)
    chip_w = max(max_x - min_x, 1e-12)
    chip_h = max(max_y - min_y, 1e-12)

    core_names, core_geom, core_types = [], [], []
    fixed_blocks = {}

    for name, w, h, x, y in blocks:
        lname = name.lower()
        if any(lname.startswith(p) for p in p_prefixes):
            ctype = 1
        elif any(lname.startswith(e) for e in e_prefixes):
            ctype = 0
        else:
            fixed_blocks[name] = float(default_fixed_power.get(name, 0.0))
            continue

        cx = (x + 0.5 * w - min_x) / chip_w
        cy = (y + 0.5 * h - min_y) / chip_h
        area = w * h
        core_names.append(name)
        core_geom.append([cx, cy, area])
        core_types.append(ctype)

    if not core_names:
        raise ValueError(
            f"No cores matched p_prefixes={p_prefixes} or e_prefixes={e_prefixes} in {flp_path}"
        )

    core_xy_area = np.array(core_geom, dtype=np.float32)
    # Normalize core areas by the largest core so values stay in [0, 1]
    core_xy_area[:, 2] /= max(float(core_xy_area[:, 2].max()), 1e-12)

    return {
        "core_names": core_names,
        "core_xy_area": core_xy_area,
        "core_types": np.array(core_types, dtype=np.int32),
        "fixed_blocks": fixed_blocks,
        "all_blocks": [b[0] for b in blocks],
    }


def run_hotspot_steady(
    hotspot_bin,
    config_path,
    flp_path,
    grid_layer_file,
    block_powers,
    work_dir="/dev/shm",
):
    """Write a 1-row .ptrace, run HotSpot steady-state, and return T_max (Kelvin or C)."""
    hotspot_bin_abs = os.path.abspath(hotspot_bin)
    config_path_abs = os.path.abspath(config_path) if config_path is not None else None
    flp_path_abs = os.path.abspath(flp_path)
    lcf_abs = os.path.abspath(grid_layer_file)
    lcf_dir = os.path.dirname(lcf_abs)
    lcf_name = os.path.basename(lcf_abs)

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        ptrace = os.path.join(tmp, "run.ptrace")
        steady = os.path.join(tmp, "run.steady")
        names = list(block_powers.keys())
        vals = [f"{block_powers[n]:.6f}" for n in names]
        with open(ptrace, "w") as f:
            f.write("\t".join(names) + "\n")
            f.write("\t".join(vals) + "\n")

        cmd = [hotspot_bin_abs]
        if config_path_abs is not None:
            cmd.extend(["-c", config_path_abs])
        cmd.extend(
            [
                "-grid_layer_file",
                lcf_name,
                "-f",
                flp_path_abs,
                "-model_type",
                "grid",
                "-p",
                ptrace,
                "-grid_steady_file",
                steady,
            ]
        )
        subprocess.run(
            cmd,
            cwd=lcf_dir,  # Runs HotSpot inside the folder containing the .lcf and .flp files
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        temps = []
        with open(steady) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        temps.append(float(parts[1]))
                    except ValueError:
                        continue
        return max(temps)


class HotSpotHeteroEnv:
    def __init__(
        self,
        core_names,
        core_xy_area,
        core_types,
        fixed_blocks,
        workload_powers_ep,
        hotspot_bin,
        config_path,
        grid_layer_file,
        flp_path,
        t_amb=318.15,
    ):
        """
        core_names:         list[str] of N core block names matching the .flp file
        core_xy_area:       (N, 3) array of normalized [x, y, area]
        core_types:         (N,) int array where 0 = E-core, 1 = P-core
        fixed_blocks:       dict[str, float] of non-core block -> constant power (W)
        workload_powers_ep: (N, 2) array where [:, 0] = E-core power, [:, 1] = P-core power
        """
        self.core_names = core_names
        self.core_geom = np.asarray(core_xy_area, dtype=np.float32)
        self.core_types = np.asarray(core_types, dtype=np.int32)
        self.fixed_blocks = fixed_blocks
        self.workload_powers = np.asarray(workload_powers_ep, dtype=np.float32)
        self.N = len(core_names)
        self.hotspot_bin = hotspot_bin
        self.config_path = config_path
        self.grid_layer_file = grid_layer_file
        self.flp_path = flp_path
        self.t_amb = t_amb
        self.cache = {}

    def action_masks(self):
        return ~self.occupied

    def _get_obs(self):
        # (N, 6) -> [x, y, area, is_p_core, assigned_power, is_occupied]
        core_feats = np.column_stack(
            [
                self.core_geom,
                self.core_types.astype(np.float32),
                self.assigned_power,
                self.occupied.astype(np.float32),
            ]
        ).astype(np.float32)

        if self.step_idx < self.N:
            cur_e, cur_p = self.workload_powers[self.step_idx]
            rem_p = self.workload_powers[self.step_idx + 1 :, 1]
            rem_mean_p = float(np.mean(rem_p)) if len(rem_p) else 0.0
            rem_max_p = float(np.max(rem_p)) if len(rem_p) else 0.0
        else:
            cur_e = cur_p = rem_mean_p = rem_max_p = 0.0

        free_mask = ~self.occupied
        free_e_frac = float(np.sum(free_mask & (self.core_types == 0))) / self.N
        free_p_frac = float(np.sum(free_mask & (self.core_types == 1))) / self.N

        global_feats = np.array(
            [cur_e, cur_p, rem_mean_p, rem_max_p, free_e_frac, free_p_frac],
            dtype=np.float32,
        )
        return core_feats, global_feats

    def reset(self):
        self.step_idx = 0
        self.occupied = np.zeros(self.N, dtype=bool)
        self.assigned_power = np.zeros(self.N, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        assert not self.occupied[action], "Invalid masked action"
        ctype = self.core_types[action]
        self.occupied[action] = True
        self.assigned_power[action] = self.workload_powers[self.step_idx, ctype]
        self.step_idx += 1
        done = self.step_idx == self.N

        reward = 0.0
        info = {}
        if done:
            key = tuple(np.round(self.assigned_power, 4))
            if key not in self.cache:
                block_powers = dict(self.fixed_blocks)
                for name, p in zip(self.core_names, self.assigned_power):
                    block_powers[name] = float(p)
                self.cache[key] = run_hotspot_steady(
                    self.hotspot_bin,
                    self.config_path,
                    self.flp_path,
                    self.grid_layer_file,
                    block_powers,
                )
            t_max = self.cache[key]
            reward = -(t_max - self.t_amb)
            info["t_max"] = t_max

        return self._get_obs(), reward, done, False, info


class MultiFlpHotSpotEnv:
    def __init__(self, scenarios, hotspot_bin, config_path, t_amb=318.15):
        """
        scenarios: list of dicts, each containing:
          - "flp_path": str
          - "flp": output of parse_flp(flp_path, ...)
          - "workload_powers_ep": (N, 2) array of [E_power, P_power]
          - "t_baseline": optional float (e.g., mean random T_max for that .flp)
        """
        self.scenarios = scenarios
        self.hotspot_bin = hotspot_bin
        self.config_path = config_path
        self.t_amb = t_amb
        self.cache = {}

    def action_masks(self):
        return ~self.occupied

    def _get_obs(self):
        core_feats = np.column_stack(
            [
                self.core_geom,
                self.core_types.astype(np.float32),
                self.assigned_power,
                self.occupied.astype(np.float32),
            ]
        ).astype(np.float32)

        if self.step_idx < self.N:
            cur_e, cur_p = self.workload_powers[self.step_idx]
            rem_p = self.workload_powers[self.step_idx + 1 :, 1]
            rem_mean_p = float(np.mean(rem_p)) if len(rem_p) else 0.0
            rem_max_p = float(np.max(rem_p)) if len(rem_p) else 0.0
        else:
            cur_e = cur_p = rem_mean_p = rem_max_p = 0.0

        free_mask = ~self.occupied
        free_e_frac = float(np.sum(free_mask & (self.core_types == 0))) / self.N
        free_p_frac = float(np.sum(free_mask & (self.core_types == 1))) / self.N

        global_feats = np.array(
            [cur_e, cur_p, rem_mean_p, rem_max_p, free_e_frac, free_p_frac],
            dtype=np.float32,
        )
        return core_feats, global_feats

    def reset(self, scenario_idx=None):
        if scenario_idx is None:
            scenario_idx = int(np.random.randint(len(self.scenarios)))
        sc = self.scenarios[scenario_idx]

        flp = sc["flp"]
        self.flp_path = sc["flp_path"]
        self.grid_layer_file = sc["grid_layer_file"]
        self.core_names = flp["core_names"]
        self.core_geom = flp["core_xy_area"]
        self.core_types = flp["core_types"]
        self.fixed_blocks = flp["fixed_blocks"]
        self.workload_powers = np.asarray(sc["workload_powers_ep"], dtype=np.float32)
        self.t_ref = sc.get("t_baseline", self.t_amb)

        self.N = len(self.core_names)
        self.step_idx = 0
        self.occupied = np.zeros(self.N, dtype=bool)
        self.assigned_power = np.zeros(self.N, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        assert not self.occupied[action], "Invalid masked action"
        ctype = self.core_types[action]
        self.occupied[action] = True
        self.assigned_power[action] = self.workload_powers[self.step_idx, ctype]
        self.step_idx += 1
        done = self.step_idx == self.N

        reward = 0.0
        info = {}
        if done:
            key = (self.flp_path, tuple(np.round(self.assigned_power, 4)))
            if key not in self.cache:
                block_powers = dict(self.fixed_blocks)
                for name, p in zip(self.core_names, self.assigned_power):
                    block_powers[name] = float(p)
                self.cache[key] = run_hotspot_steady(
                    self.hotspot_bin,
                    self.config_path,
                    self.flp_path,
                    self.grid_layer_file,
                    block_powers,
                )
            t_max = self.cache[key]
            reward = -(t_max - self.t_ref)
            info["t_max"] = t_max

        return self._get_obs(), reward, done, False, info


def estimate_t_baseline(
    flp,
    workload_powers_ep,
    hotspot_bin,
    config_path,
    flp_path,
    grid_layer_file,
    n_samples=10,
    seed=42,
    max_workers=None,
):
    rng = np.random.default_rng(seed)
    core_names = flp["core_names"]
    core_types = flp["core_types"]
    fixed_blocks = flp["fixed_blocks"]
    n = len(core_names)
    wl = np.asarray(workload_powers_ep, dtype=np.float32)

    jobs = []
    for _ in range(n_samples):
        perm = rng.permutation(n)
        block_powers = dict(fixed_blocks)
        for wl_idx, core_idx in enumerate(perm):
            ctype = core_types[core_idx]
            block_powers[core_names[core_idx]] = float(wl[wl_idx, ctype])
        jobs.append(block_powers)

    with ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as pool:
        samples = list(
            pool.map(
                lambda bp: run_hotspot_steady(
                    hotspot_bin, config_path, flp_path, grid_layer_file, bp
                ),
                jobs,
            )
        )
    return float(np.mean(samples))


# def estimate_t_baseline(
#     flp,
#     workload_powers_ep,
#     hotspot_bin,
#     config_path,
#     flp_path,
#     grid_layer_file,
#     n_samples=10,
#     seed=42,
# ):
#     """Estimate mean random-mapping T_max for a floorplan to normalize RL rewards."""
#     rng = np.random.default_rng(seed)
#     core_names = flp["core_names"]
#     core_types = flp["core_types"]
#     fixed_blocks = flp["fixed_blocks"]
#     n = len(core_names)
#     wl = np.asarray(workload_powers_ep, dtype=np.float32)
#
#     samples = []
#     for _ in range(n_samples):
#         perm = rng.permutation(n)
#         block_powers = dict(fixed_blocks)
#         for wl_idx, core_idx in enumerate(perm):
#             ctype = core_types[core_idx]
#             block_powers[core_names[core_idx]] = float(wl[wl_idx, ctype])
#         samples.append(
#             run_hotspot_steady(
#                 hotspot_bin, config_path, flp_path, grid_layer_file, block_powers
#             )
#         )
#
#     return float(np.mean(samples))


# flp = parse_flp(
#     "fp_001_layer_0.flp",
#     p_prefixes=("Core_0", "Core_1"),
#     e_prefixes=("Core",),
#     default_fixed_power={},
# )
# N = len(flp["core_names"])


# env = HotSpotHeteroEnv(
#     core_names=flp["core_names"],
#     core_xy_area=flp["core_xy_area"],
#     core_types=flp["core_types"],
#     fixed_blocks=flp["fixed_blocks"],
#     workload_powers_ep=workload_powers_ep,
#     hotspot_bin="extern/HotSpot/hotspot",
#     config_path="hotspot.config",
#     grid_layer_file="fp_001.lcf",
#     flp_path="fp_001_layer_0.flp",
# )


workload_powers_ep = np.array(
    [
        [2.1, 9.5],
        [3.4, 14.2],
        [1.2, 4.0],
        [2.8, 11.0],
    ],
    dtype=np.float32,
)
WORKLOAD_BANK = [
    [0.4, 1.2],
    [1.1, 3.8],
    [1.5, 4.2],
    [1.8, 5.5],
    [2.1, 6.8],
    [2.3, 7.4],
    [2.6, 8.9],
    [2.9, 10.5],
    [3.1, 11.2],
    [3.4, 13.8],
    [3.6, 14.5],
    [3.8, 15.2],
    [4.0, 16.5],
    [4.2, 18.0],
    [4.5, 19.5],
    [4.8, 21.0],
]


def sample_workloads(workload_bank, n_cores, rng=None, replace=True):
    """Randomly select `n_cores` [E_power, P_power] rows from `workload_bank`."""
    rng = rng or np.random.default_rng()
    bank = np.asarray(workload_bank, dtype=np.float32)
    idxs = rng.choice(len(bank), size=n_cores, replace=replace or (n_cores > len(bank)))
    return bank[idxs]


hotspot_bin = "./extern/HotSpot/hotspot"
config_path = None
rng = np.random.default_rng(42)

floorplan_specs = [
    (
        "Sample_dataset/2D_floorplans/Style_1/fp_001/fp_001_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_1/fp_001/fp_001.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_1/fp_002/fp_002_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_1/fp_002/fp_002.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_1/fp_003/fp_003_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_1/fp_003/fp_003.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_2/fp_001/fp_001_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_2/fp_001/fp_001.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_2/fp_002/fp_002_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_2/fp_002/fp_002.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_2/fp_003/fp_003_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_2/fp_003/fp_003.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_2/fp_004/fp_004_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_2/fp_004/fp_004.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_2/fp_005/fp_005_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_2/fp_005/fp_005.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_3/fp_001/fp_001_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_3/fp_001/fp_001.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_3/fp_002/fp_002_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_3/fp_002/fp_002.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_3/fp_003/fp_003_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_3/fp_003/fp_003.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_3/fp_004/fp_004_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_3/fp_004/fp_004.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_3/fp_005/fp_005_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_3/fp_005/fp_005.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_4/fp_001/fp_001_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_4/fp_001/fp_001.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_4/fp_002/fp_002_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_4/fp_002/fp_002.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_4/fp_003/fp_003_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_4/fp_003/fp_003.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_4/fp_004/fp_004_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_4/fp_004/fp_004.lcf",
    ),
    (
        "Sample_dataset/2D_floorplans/Style_4/fp_005/fp_005_layer_0.flp",
        "Sample_dataset/2D_floorplans/Style_4/fp_005/fp_005.lcf",
    ),
]

scenarios = []
for flp_path, lcf_path in floorplan_specs:
    flp = parse_flp(
        flp_path,
        p_prefixes=("core_0_", "core_1_"),
        e_prefixes=("core_"),
    )
    n_cores = len(flp["core_names"])
    workload_powers_ep = sample_workloads(WORKLOAD_BANK, n_cores, rng=rng)
    t_baseline = estimate_t_baseline(
        flp=flp,
        workload_powers_ep=workload_powers_ep,
        hotspot_bin=hotspot_bin,
        config_path=config_path,
        flp_path=flp_path,
        grid_layer_file=lcf_path,
    )
    scenarios.append(
        {
            "flp_path": flp_path,
            "grid_layer_file": lcf_path,
            "flp": flp,
            "workload_powers_ep": workload_powers_ep,
            "t_baseline": t_baseline,
        }
    )
n_max = max(len(s["flp"]["core_names"]) for s in scenarios)
env = MultiFlpHotSpotEnv(scenarios, hotspot_bin, config_path)

policy = PointerActorCritic(
    core_dim=6,
    global_dim=6,
    d_model=128,
    nhead=4,
    num_layers=2,
)

train_masked_ppo(
    env=env,
    policy=policy,
    n_max=n_max,
    total_updates=200,
    steps_per_rollout=512,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

torch.save(policy.state_dict(), "hotspot_mapper_policy.pt")
