#set text(size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[
    Project Status Report: Thermal Workload Mapper
  ]
  #v(0.4em)
]

#v(0.8em)

= Implemented Components

The end-to-end reinforcement learning pipeline for mapping $N$ workloads 1-to-1 onto $N$ heterogeneous CPU cores (Performance *P-cores* and Efficiency *E-cores*) has been implemented:

- *Floorplan Parser (`parse_flp`):* Parses HotSpot 7.0 `.flp` files, normalizes core center coordinates $(c_x, c_y)$ and areas to $[0, 1]$, tags P-cores (`core_0_`, `core_1_`) vs. E-cores (`core_`), and separates non-core blocks into `fixed_blocks`.
- *HotSpot 7.0 Grid-Mode Runner (`run_hotspot_steady`):* Writes 1-row `.ptrace` files in `/dev/shm` (RAM disk), invokes HotSpot 7.0 in steady-state grid mode (`-model_type grid`, `-grid_layer_file`, `-grid_steady_file`), and extracts peak temperature $T_max$.
- *Parallel Baseline Estimator (`estimate_t_baseline`):* Uses `ThreadPoolExecutor` to evaluate 10 random valid 1-to-1 mappings per floorplan in parallel and compute `t_baseline` for reward normalization across different chip layouts.
- *Multi-Floorplan Environment (`MultiFlpHotSpotEnv`):* Gym-free environment sampling across 18 floorplan/layer pairs (`Style_1`–`Style_4`), building `(N, 6)` per-core tokens and `(6,)` global workload/slot features, enforcing `action_masks()`, and caching `(flp_path, power_tuple) -> T_max`.
- *Set-Transformer Pointer Policy (`PointerActorCritic`):* $N$-agnostic, permutation-equivariant actor-critic built with a 2-layer, 4-head `TransformerEncoder` (`d_model=128`), a `q · k_j` pointer head with `-1e8` masking on occupied/padded slots, and a masked mean-pooled critic head.
- *Masked PPO + GAE Trainer (`train_masked_ppo`):* Trains with `gamma=1.0`, `gae_lambda=0.95`, `clip_eps=0.2`, `pad_step`/`pad_mask` for mixed core counts, and saves `best_hotspot_mapper.pt` and `hotspot_mapper_policy.pt`.

= Testing & Verification Approach

Proper Verification and testing approach remains.

// == Verification Approach
// - *Random Baseline Comparison:* Compare the deterministic trained policy's $T_max$ against each floorplan's `t_baseline` (mean $T_max$ over random valid mappings) across all 18 floorplans.
// - *Zero-Shot Generalization:* Evaluate the saved checkpoint on unseen `.flp` floorplans and varying core counts to verify that the Set-Transformer pointer policy transfers without retraining.
// - *Constraint & Spatial Sanity Checks:* Verify 100% valid 1-to-1 mappings via `action_masks()` and confirm that high-power workloads are distributed away from adjacent central P-cores.

// == Test Results
// #block(fill: luma(245), stroke: 0.5pt + luma(200), inset: 10pt, radius: 4pt, width: 100%)[
//   *[Pending — Training Currently in Progress]* \
//   Insert observed results once training completes:
//   - *Training Duration / Updates Completed:* `[TBD]`
//   - *Mean $T_max$ vs. Random `t_baseline` (per style/floorplan):* `[TBD]`
//   - *Unseen Floorplan Generalization $T_max$:* `[TBD]`
// ]

= Plan Adherence & Deviations

- *On Plan:* The core RL formulation (sequential assignment MDP with action masking), heterogeneous P/E core modeling, HotSpot 7.0 steady-state integration, and multi-floorplan training pipeline proceeded as planned.
- *Deviation (Workload Generation):* A proper workload estimator based on instructions, CPU specifications, and workload analysis could not be integrated within the current scope. As a workaround, hardcoded synthetic `[E_power, P_power]` pairs (`WORKLOAD_BANK`, 16 profiles from `[0.4, 1.2]` W to `[4.8, 21.0]` W) are sampled via `sample_workloads()`.

= Learnings & Future Scope

== Key Learnings
- Breaking an $N!$ permutation problem into an $N$-step sequential assignment with `action_masks()` keeps the action space linear ($N$) and prevents wasted HotSpot evaluations on invalid moves.
- Replacing a fixed flat MLP with a Set-Transformer + Pointer head (`q · k_j`) allows one policy to handle varying core counts and `.flp` layouts without positional memorization.
- Using `/dev/shm` temporary directories, memoization caching, and `ThreadPoolExecutor` for baseline estimation significantly reduces HotSpot subprocess overhead.

== Future Scope
- *Proper Workload Estimator:* Build an accurate workload power and runtime estimator driven by instruction mixes, CPU microarchitectural specifications (P-core vs. E-core), and static/dynamic workload analysis to replace `WORKLOAD_BANK`.
- *Dynamic Workload Scheduling:* Extend from static steady-state mapping to dynamic, time-aware scheduling based on the exact execution duration of each workload, using HotSpot 7.0 transient thermal simulation (`.ptrace` time steps) and accounting for thread migration overhead.
- *Thermal–Performance Co-Optimization:* Incorporate an IPC/throughput term into the reward alongside `-(T_max - t_baseline)` to balance peak temperature reduction against P-core utilization.

= LLM Assistance Disclosure

LLM assistance was used during this project for the following tasks:
- *RL Formulation & Architecture Design:* Designing the sequential assignment MDP, action-masking strategy, reward normalization (`t_baseline`), and the Set-Transformer `PointerActorCritic` architecture.
- *Code Drafting & Explanation:* Drafting initial Python templates for `parse_flp`, `HotSpotHeteroEnv`/`MultiFlpHotSpotEnv`, `PointerActorCritic`, `estimate_t_baseline`, and the PyTorch Masked PPO + GAE training loop, as well as explaining PyTorch/RL concepts (GAE, self/cross-attention, batching, and checkpointing).
- *Report Structuring:* Organizing and formatting this project status report in Markdown and Typst.

All HotSpot 7.0 grid-mode (`.lcf`) adaptations, floorplan dataset integration (`Style_1`–`Style_4`), `WORKLOAD_BANK` definitions, `ThreadPoolExecutor` parallelization, and training execution were carried out by the mee.
