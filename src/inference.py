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


import csv
import time

import numpy as np


def eval_random_and_greedy(
    flp, lcf_path, flp_path, workloads, hotspot_bin, config_path=None, n_rand=20, seed=0
):
    rng = np.random.default_rng(seed)
    core_names = flp["core_names"]
    core_types = flp["core_types"]
    fixed_blocks = flp["fixed_blocks"]
    N = len(core_names)

    # 1. Random baseline (mean & best of n_rand)
    rand_temps = []
    for _ in range(n_rand):
        perm = rng.permutation(N)
        bp = dict(fixed_blocks)
        for wl_i, c_i in enumerate(perm):
            bp[core_names[c_i]] = float(workloads[wl_i, core_types[c_i]])
        rand_temps.append(
            run_hotspot_steady(hotspot_bin, config_path, flp_path, lcf_path, bp)
        )

    # 2. Greedy heuristic: largest (P_P - P_E) workloads -> E-cores first
    e_cores = [i for i, t in enumerate(core_types) if t == 0]
    p_cores = [i for i, t in enumerate(core_types) if t == 1]
    order = np.argsort(-(workloads[:, 1] - workloads[:, 0]))
    bp_greedy = dict(fixed_blocks)
    for wl_i, core_i in zip(order, e_cores + p_cores):
        bp_greedy[core_names[core_i]] = float(workloads[wl_i, core_types[core_i]])
    greedy_t = run_hotspot_steady(
        hotspot_bin, config_path, flp_path, lcf_path, bp_greedy
    )

    return float(np.mean(rand_temps)), float(np.min(rand_temps)), greedy_t


def run_benchmark(
    ckpt_path,
    floorplan_specs,
    workload_bank,
    hotspot_bin="./extern/HotSpot/hotspot",
    n_seeds=5,
):
    rng = np.random.default_rng(123)
    rows = []

    for flp_path, lcf_path in floorplan_specs:
        flp = parse_flp(
            flp_path, p_prefixes=("core_0_", "core_1_"), e_prefixes=("core_",)
        )
        n = len(flp["core_names"])
        bank = np.asarray(workload_bank, dtype=np.float32)

        rl_list, rand_mean_list, rand_min_list, greedy_list, ms_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for s in range(n_seeds):
            idxs = rng.choice(len(bank), size=n, replace=True)
            wl = bank[idxs]

            t0 = time.perf_counter()
            _, rl_t = infer_mapping(
                ckpt_path, flp_path, lcf_path, wl, hotspot_bin=hotspot_bin
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0

            r_mean, r_min, g_t = eval_random_and_greedy(
                flp, lcf_path, flp_path, wl, hotspot_bin, n_rand=20, seed=s
            )
            rl_list.append(rl_t)
            rand_mean_list.append(r_mean)
            rand_min_list.append(r_min)
            greedy_list.append(g_t)
            ms_list.append(dt_ms)

        row = {
            "flp": flp_path.split("/")[-1],
            "cores": n,
            "rand_mean": np.mean(rand_mean_list),
            "rand_min": np.mean(rand_min_list),
            "greedy": np.mean(greedy_list),
            "rl": np.mean(rl_list),
            "gain_vs_rand": np.mean(rand_mean_list) - np.mean(rl_list),
            "gain_vs_greedy": np.mean(greedy_list) - np.mean(rl_list),
        }
        rows.append(row)
        print(
            f"{row['flp']:20s} | N={n:2d} | Rand={row['rand_mean']:.2f} K | "
            f"Greedy={row['greedy']:.2f} K | RL={row['rl']:.2f} K | "
            f"Gain vs Rand={row['gain_vs_rand']:+.2f} K | vs Greedy={row['gain_vs_greedy']:+.2f} K"
        )

    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


WORKLOAD_BANK = [
    # Light / Idle / Memory-bound (low P-core penalty)
    [0.35, 0.90],  # Idle / background daemon
    [0.55, 1.45],  # Pointer-chasing / memory-bound
    [0.75, 2.10],  # Streaming I/O / cache-miss heavy
    [0.95, 2.80],  # Lightweight web / event loop
    [1.15, 3.50],  # Light compression (zstd -1)
    [1.30, 4.10],  # Database lookup / B-tree scan
    # Moderate / Mixed integer & branch-heavy
    [1.55, 5.00],  # Compiler frontend / parsing
    [1.80, 5.90],  # JSON/protobuf serialization
    [2.05, 6.85],  # Integer sorting / graph traversal
    [2.25, 7.60],  # WebAssembly / JS JIT workload
    [2.50, 8.50],  # Video decode (mixed scalar/SIMD)
    [2.75, 9.40],  # Encryption / AES-GCM stream
    # Compute-intensive (moderate-high P-core spike)
    [3.00, 10.60],  # Sparse linear algebra
    [3.20, 11.50],  # Physics / collision step
    [3.45, 12.60],  # Ray-tracing / BVH traversal
    [3.65, 13.80],  # High-ratio compression (xz)
    [3.90, 14.90],  # Audio/video encode (x265)
    [4.10, 16.00],  # ML inference (INT8 GEMM)
    # Heavy FP / SIMD / Thermal-stress (steepest P-core spike)
    [4.30, 17.20],  # Dense FP32 matrix multiply
    [4.50, 18.40],  # FFT / signal processing
    [4.70, 19.60],  # Molecular dynamics / stencil
    [4.90, 20.80],  # Monte Carlo simulation
    [5.10, 22.00],  # AVX2/AVX-512 GEMM kernel
    [5.35, 23.50],  # Prime / cryptographic stress
]


def run_all_tests(
    ckpt_path,
    floorplan_specs,
    workload_bank,
    hotspot_bin="./extern/HotSpot/hotspot",
    n_seeds=5,
):
    rng = np.random.default_rng(123)
    bank = np.asarray(workload_bank, dtype=np.float32)
    bank_sorted = bank[np.argsort(bank[:, 1])]  # sorted cool -> hot

    all_rows = []

    for flp_path, lcf_path in floorplan_specs:
        flp = parse_flp(
            flp_path, p_prefixes=("core_0_", "core_1_"), e_prefixes=("core_",)
        )
        n = len(flp["core_names"])
        flp_name = "/".join(flp_path.split("/")[-2:])  # e.g. fp_001/fp_001_layer_0.flp

        # Build 4 distinct test suites for this floorplan
        suites = {
            "mixed": [
                bank[rng.choice(len(bank), size=n, replace=True)]
                for _ in range(n_seeds)
            ],
            "skewed": [
                np.vstack(
                    [
                        bank_sorted[-max(2, n // 4) :][
                            rng.choice(
                                max(2, n // 4), size=max(2, n // 4), replace=True
                            )
                        ],
                        bank_sorted[: len(bank_sorted) // 2][
                            rng.choice(
                                len(bank_sorted) // 2,
                                size=n - max(2, n // 4),
                                replace=True,
                            )
                        ],
                    ]
                )[rng.permutation(n)]
                for _ in range(n_seeds)
            ],
            "heavy_stress": [
                bank_sorted[len(bank_sorted) // 2 :][
                    rng.choice(
                        len(bank_sorted) - len(bank_sorted) // 2, size=n, replace=True
                    )
                ]
                for _ in range(n_seeds)
            ],
        }
        # Suite 4: fixed workload set in 5 different arrival orders
        base_wl = bank[rng.choice(len(bank), size=n, replace=True)]
        suites["order_invariance"] = [
            base_wl[rng.permutation(n)] for _ in range(n_seeds)
        ]

        for suite_name, wl_list in suites.items():
            rl_t, rand_mean_t, rand_min_t, greedy_t, ms_t = [], [], [], [], []
            for s, wl in enumerate(wl_list):
                t0 = time.perf_counter()
                _, t_rl = infer_mapping(
                    ckpt_path, flp_path, lcf_path, wl, hotspot_bin=hotspot_bin
                )
                ms_t.append((time.perf_counter() - t0) * 1000.0)

                r_mean, r_min, g_t = eval_random_and_greedy(
                    flp, lcf_path, flp_path, wl, hotspot_bin, n_rand=20, seed=s
                )
                rl_t.append(t_rl)
                rand_mean_t.append(r_mean)
                rand_min_t.append(r_min)
                greedy_t.append(g_t)

            row = {
                "flp": flp_name,
                "suite": suite_name,
                "cores": n,
                "rand_mean": np.mean(rand_mean_t),
                "rand_min": np.mean(rand_min_t),
                "greedy": np.mean(greedy_t),
                "rl_mean": np.mean(rl_t),
                "rl_std": np.std(rl_t),
                "gain_vs_rand": np.mean(rand_mean_t) - np.mean(rl_t),
                "gain_vs_greedy": np.mean(greedy_t) - np.mean(rl_t),
                "infer_ms": np.mean(ms_t),
            }
            all_rows.append(row)
            print(
                f"{flp_name:28s} | {suite_name:16s} | Rand={row['rand_mean']:.2f} K | "
                f"Greedy={row['greedy']:.2f} K | RL={row['rl_mean']:.2f}±{row['rl_std']:.2f} K | "
                f"Gain(Rand)={row['gain_vs_rand']:+.2f} K | Gain(Greedy)={row['gain_vs_greedy']:+.2f} K"
            )

    with open("all_test_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print("\nSaved all results to all_test_results.csv")


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

if __name__ == "__main__":
    run_all_tests(
        ckpt_path="best_hotspot_mapper.pt",
        floorplan_specs=floorplan_specs,
        workload_bank=WORKLOAD_BANK,
        hotspot_bin="./extern/HotSpot/hotspot",
        n_seeds=5,
    )
    # run_benchmark(
    #     ckpt_path="best_hotspot_mapper.pt",
    #     floorplan_specs=[
    #         (
    #             "Sample_dataset/2D_floorplans/Style_1/fp_001/fp_001_layer_0.flp",
    #             "Sample_dataset/2D_floorplans/Style_1/fp_001/fp_001.lcf",
    #         ),
    #     ],
    #     workload_bank=WORKLOAD_BANK,
    #     hotspot_bin="./extern/HotSpot/hotspot",
    #     n_seeds=10,
    # )

if __name__ != "__main__":
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
