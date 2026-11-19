import math
import os
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PointerActorCritic(nn.Module):
    def __init__(self, core_dim=6, global_dim=6, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.core_embed = nn.Sequential(
            nn.Linear(core_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.query_proj = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, core_feats, global_feats, action_mask, pad_mask=None):
        h = self.core_embed(core_feats)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        q = self.query_proj(global_feats).unsqueeze(1)
        k = self.key_proj(h)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
        logits = logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=logits)
        return dist, None  # Critic not needed for inference


def parse_flp(
    flp_path,
    p_prefixes=("core_0_", "core_1_"),
    e_prefixes=("core_",),
    default_fixed_power=None,
):
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
            blocks.append((parts[0], *map(float, parts[1:5])))

    min_x = min(x for _, _, _, x, _ in blocks)
    min_y = min(y for _, _, _, _, y in blocks)
    max_x = max(x + w for _, w, _, x, _ in blocks)
    max_y = max(y + h for _, _, h, x, y in blocks)
    chip_w, chip_h = max(max_x - min_x, 1e-12), max(max_y - min_y, 1e-12)

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
        core_names.append(name)
        core_geom.append(
            [(x + 0.5 * w - min_x) / chip_w, (y + 0.5 * h - min_y) / chip_h, w * h]
        )
        core_types.append(ctype)

    core_xy_area = np.array(core_geom, dtype=np.float32)
    core_xy_area[:, 2] /= max(float(core_xy_area[:, 2].max()), 1e-12)
    return {
        "core_names": core_names,
        "core_xy_area": core_xy_area,
        "core_types": np.array(core_types, dtype=np.int32),
        "fixed_blocks": fixed_blocks,
    }


def run_hotspot_steady(
    hotspot_bin,
    config_path,
    flp_path,
    grid_layer_file,
    block_powers,
    work_dir="/dev/shm",
):
    lcf_abs = os.path.abspath(grid_layer_file)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        ptrace = os.path.join(tmp, "run.ptrace")
        steady = os.path.join(tmp, "run.steady")
        names = list(block_powers.keys())
        with open(ptrace, "w") as f:
            f.write("\t".join(names) + "\n")
            f.write("\t".join(f"{block_powers[n]:.6f}" for n in names) + "\n")

        cmd = [os.path.abspath(hotspot_bin)]
        if config_path is not None:
            cmd.extend(["-c", os.path.abspath(config_path)])
        cmd.extend(
            [
                "-grid_layer_file",
                os.path.basename(lcf_abs),
                "-f",
                os.path.abspath(flp_path),
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
            cwd=os.path.dirname(lcf_abs),
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


@torch.no_grad()
def infer_mapping(
    ckpt_path,
    flp_path,
    lcf_path,
    workload_powers_ep,
    hotspot_bin="./extern/HotSpot/hotspot",
    config_path=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load trained weights
    policy = PointerActorCritic(
        core_dim=6, global_dim=6, d_model=128, nhead=4, num_layers=2
    ).to(device)
    policy.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    policy.eval()

    # 2. Parse floorplan
    flp = parse_flp(flp_path)
    core_names = flp["core_names"]
    core_geom = flp["core_xy_area"]
    core_types = flp["core_types"]
    wl = np.asarray(workload_powers_ep, dtype=np.float32)
    N = len(core_names)
    assert len(wl) == N, f"Expected {N} workloads, got {len(wl)}"

    occupied = np.zeros(N, dtype=bool)
    assigned_power = np.zeros(N, dtype=np.float32)
    mapping = []

    # 3. Greedy step-by-step assignment
    for t in range(N):
        core_feats = np.column_stack(
            [
                core_geom,
                core_types.astype(np.float32),
                assigned_power,
                occupied.astype(np.float32),
            ]
        ).astype(np.float32)

        cur_e, cur_p = wl[t]
        rem_p = wl[t + 1 :, 1]
        rem_mean_p = float(np.mean(rem_p)) if len(rem_p) else 0.0
        rem_max_p = float(np.max(rem_p)) if len(rem_p) else 0.0
        free_mask = ~occupied
        free_e_frac = float(np.sum(free_mask & (core_types == 0))) / N
        free_p_frac = float(np.sum(free_mask & (core_types == 1))) / N

        global_feats = np.array(
            [cur_e, cur_p, rem_mean_p, rem_max_p, free_e_frac, free_p_frac],
            dtype=np.float32,
        )

        c_t = torch.tensor(core_feats, dtype=torch.float32, device=device).unsqueeze(0)
        g_t = torch.tensor(global_feats, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        m_t = torch.tensor(free_mask, dtype=torch.bool, device=device).unsqueeze(0)

        dist, _ = policy(c_t, g_t, m_t)
        action = int(dist.probs.argmax(dim=-1).item())  # deterministic best valid core

        occupied[action] = True
        assigned_power[action] = wl[t, core_types[action]]
        mapping.append(
            (
                t,
                core_names[action],
                "P" if core_types[action] == 1 else "E",
                float(assigned_power[action]),
            )
        )

    # 4. Verify final steady-state T_max with HotSpot 7.0
    block_powers = dict(flp["fixed_blocks"])
    for name, p in zip(core_names, assigned_power):
        block_powers[name] = float(p)
    t_max = run_hotspot_steady(
        hotspot_bin, config_path, flp_path, lcf_path, block_powers
    )

    return mapping, t_max


if __name__ == "__main__":
    # Example usage on one floorplan
    flp_path = "Sample_dataset/2D_floorplans/Style_3/fp_001/fp_001_layer_0.flp"
    lcf_path = "Sample_dataset/2D_floorplans/Style_3/fp_001/fp_001.lcf"
    flp_info = parse_flp(flp_path)
    n = len(flp_info["core_names"])

    # Replace with your N x 2 [E_power, P_power] workload table
    rng = np.random.default_rng(0)
    workloads = np.hstack(
        [
            rng.uniform(2.0, 4.0, size=(n, 1)),
            rng.uniform(15.0, 28.0, size=(n, 1)),
        ]
    ).astype(np.float32)

    mapping, t_max = infer_mapping(
        ckpt_path="best_hotspot_mapper.pt",
        flp_path=flp_path,
        lcf_path=lcf_path,
        workload_powers_ep=workloads,
    )
    for wl_idx, core_name, ctype, power in mapping:
        e_pow, p_pow = workloads[wl_idx]
        print(
            f"Workload {wl_idx:2d} [E={e_pow:.2f} W, P={p_pow:.2f} W] "
            f"-> {core_name} ({ctype}-core, {power:.2f} W)"
        )
    print(f"Final HotSpot T_max: {t_max:.2f} K")
