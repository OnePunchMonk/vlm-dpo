# VLM-DPO: Research Execution Plan

## Project Summary

**Goal**: Align video diffusion models using Vision-Language Models as automated preference oracles, eliminating the need for human preference annotations.

**Core Idea**: DPO/RLHF transformed LLMs, but preference-based alignment of video diffusion remains unexplored because human video preference data is impractical to collect at scale. We use InternVL-U (4B) as a free, scalable preference judge — scoring generated videos across multiple quality dimensions — then train LoRA-DPO on Wan2.1-T2V-1.3B.

**Three Key Contributions**:
1. **VLM Preference Distillation** — Multi-aspect VLM prompting strategies that decompose video quality into scoreable dimensions, replacing human annotation entirely
2. **Cross-Modal DPO Transfer** — Curriculum strategy: train image DPO on Flux.2, then transfer LoRA weights to video DiT (Wan2.1)
3. **Multi-Aspect Reward Decomposition** — Decomposed DPO loss `r = α_p·r_p + α_t·r_t + α_v·r_v + α_m·r_m`, studying which reward dimension matters most

---

## Phase 1 — Validate the Oracle (Exp 1)

**Timeline**: ~1 week

**Objective**: Establish that InternVL-U is a reliable preference judge by measuring agreement with human annotators.

**Why first**: If VLM scores don't correlate with human preferences, the entire premise collapses. This experiment gates all subsequent work.

### Steps
1. Generate 500 video pairs using Wan2.1 with diverse prompts (varied subjects, actions, styles)
2. Score each pair with InternVL-U using all 3 strategies:
   - **Holistic**: Single 1-10 quality score
   - **Multi-aspect**: Per-dimension scores (prompt adherence, temporal consistency, visual quality, motion naturalness)
   - **CoT**: Chain-of-thought reasoning before scoring
3. Build HTML comparison interface for human annotation (already implemented in `vlm_dpo/evaluation/human_eval.py`)
4. Collect 500 human A/B preference annotations (3 annotators per pair for inter-rater reliability)
5. Compute Cohen's κ between VLM and human majority vote

### Success Criteria
- Cohen's κ > 0.6 (substantial agreement) for at least one scoring strategy
- Multi-aspect or CoT strategy outperforms holistic
- Per-dimension agreement analysis: identify which aspects VLMs judge best

### Key Config
- `configs/exp1_vlm_agreement.yaml`
- 500 pairs, all 3 scoring strategies ablated

### Outputs
- `outputs/exp1/vlm_scores_{strategy}.jsonl` — VLM preference labels per strategy
- `outputs/exp1/human_eval/` — HTML interface + collected annotations
- `outputs/exp1/metrics.json` — Cohen's κ, agreement rate, Spearman ρ per strategy

---

## Phase 2 — Main Result: Video DPO on Wan2.1 (Exp 3)

**Timeline**: ~2 weeks

**Objective**: Train LoRA-DPO on Wan2.1-1.3B using VLM preference data and demonstrate measurable quality improvement.

**This is the core contribution** — the first DPO alignment of a video diffusion model using automated VLM preferences.

### Steps
1. Generate 5K preference pairs: for each prompt, generate 2 videos with different seeds, score with InternVL-U (multi-aspect), assign winner/loser
2. Train LoRA-DPO on Wan2.1-1.3B:
   - Baseline config: β=0.1, rank=16, α=32, 2K steps, effective batch=8
   - Frozen reference model (deep copy of base Wan2.1)
3. Run ablation grid:
   - **β (DPO temperature)**: [0.01, 0.05, 0.1, 0.2, 0.5] — controls preference sharpness
   - **LoRA rank**: [4, 8, 16, 32, 64] — model capacity
   - **Data scale**: [500, 1K, 2K, 5K] — data efficiency curve
4. Evaluate all checkpoints:
   - FVD (Frechet Video Distance) — distributional quality
   - CLIP score — prompt-content alignment
   - VBench (subject consistency, motion smoothness, aesthetic quality) — per-dimension quality
   - Human preference evaluation on 100 sample A/B comparisons (aligned vs base)

### Baselines
- **Base Wan2.1** (no alignment) — lower bound
- **SFT-only** — train on "winner" videos only, no preference signal
- **Random preference** — DPO with shuffled winner/loser labels — ablation control

### Success Criteria
- FVD improvement ≥ 10% over base Wan2.1
- CLIP score improvement ≥ 5%
- VBench improvement on at least 2/3 dimensions
- Human evaluators prefer aligned model ≥ 60% of the time

### Key Config
- `configs/exp3_video_dpo.yaml`
- 5K pairs, 2K training steps, ablation grids defined

### Outputs
- `data/exp3_video_pairs/` — 5K preference pairs with metadata
- `outputs/exp3/checkpoints/` — trained LoRA weights
- `outputs/exp3/ablations/` — per-ablation results
- `outputs/exp3/metrics.json` — consolidated metrics

---

## Phase 3 — Cross-Modal Transfer (Exp 2 + Exp 4)

**Timeline**: ~1.5 weeks

**Objective**: Test whether image DPO LoRA weights provide a better initialization for video DPO than training from scratch.

**Novel angle**: Image preference data is 10x cheaper to generate (no temporal dimension). If image→video transfer works, it dramatically reduces the cost of video alignment.

### Steps

#### Exp 2: Image DPO on Flux.2 (prerequisite)
1. Generate 10K image preference pairs using Flux.2-dev + InternVL-U scoring
2. Train LoRA-DPO on Flux.2-dev (5K steps, rank=16)
3. Evaluate: FID, CLIP score, human preference
4. Save best checkpoint to `outputs/exp2/best_checkpoint/`

#### Exp 4: Cross-Modal Transfer
1. Load Exp 2 LoRA weights as initialization for Wan2.1 LoRA
2. Transfer using name-matching strategy (matching attention layer names across DiT architectures)
3. Freeze transferred layers for 200 steps, then unfreeze for full fine-tuning
4. Train video DPO for 2K steps on same Exp 3 data
5. Compare against Exp 3 from-scratch baseline

### Ablations
- **Mapping strategy**: name_match vs position_match
- **Freeze duration**: [0, 100, 200, 500] steps
- **Transfer subset**: all layers vs attention-only vs value-projection-only

### Success Criteria
- Transfer model matches or exceeds from-scratch baseline within fewer training steps
- Faster convergence (lower loss at step 500 vs from-scratch)
- Transfer compatibility rate > 50% of LoRA parameters

### Key Configs
- `configs/exp2_image_dpo.yaml` — image DPO training
- `configs/exp4_cross_modal.yaml` — transfer experiment

### Outputs
- `outputs/exp2/` — image DPO checkpoint + metrics
- `outputs/exp4/` — transfer results + comparison with Exp 3
- `outputs/exp4/transfer_analysis.json` — layer compatibility report

---

## Phase 4 — Reward Decomposition (Exp 5)

**Timeline**: ~1 week

**Objective**: Identify which quality dimension drives the most improvement in video DPO, and whether decomposed multi-aspect scoring outperforms holistic scoring.

### Steps
1. Using the same 5K pairs from Exp 3, retrain with different reward weight configurations:

| Config | α_prompt | α_temporal | α_visual | α_motion | Purpose |
|:-------|:---------|:-----------|:---------|:---------|:--------|
| prompt-only | 1.0 | 0.0 | 0.0 | 0.0 | Isolate prompt adherence |
| temporal-only | 0.0 | 1.0 | 0.0 | 0.0 | Isolate temporal consistency |
| visual-only | 0.0 | 0.0 | 1.0 | 0.0 | Isolate visual quality |
| motion-only | 0.0 | 0.0 | 0.0 | 1.0 | Isolate motion naturalness |
| equal | 0.25 | 0.25 | 0.25 | 0.25 | Uniform combination |
| balanced | 0.3 | 0.3 | 0.2 | 0.2 | Our proposed weighting |

2. Additionally compare: holistic (single score) vs multi-aspect (decomposed) scoring strategy
3. Evaluate all configurations on FVD, CLIP score, VBench

### Analysis
- Which single dimension provides the largest improvement?
- Does multi-aspect outperform holistic? By how much?
- Is the balanced weighting optimal, or does one dimension dominate?
- Cross-metric analysis: does optimizing for temporal consistency hurt visual quality (tradeoffs)?

### Success Criteria
- Clear ranking of dimension importance
- Multi-aspect outperforms holistic on at least 2/3 metrics
- Balanced weighting within 5% of best single-dimension on that dimension's target metric

### Key Config
- `configs/exp5_multi_aspect.yaml`

### Outputs
- `outputs/exp5/ablations/` — per-weight-config results
- `outputs/exp5/metrics.json` — consolidated comparison table

---

## Hardware Requirements

| Component | VRAM | Time |
|:----------|:-----|:-----|
| InternVL-U preference scoring | ~10 GB | ~2 sec/pair |
| Wan2.1 video generation (16 frames) | ~16 GB | ~15 sec/video |
| Flux.2 image generation | ~20 GB (FP8) | ~5 sec/image |
| LoRA DPO training (Wan2.1) | ~16 GB | ~8-12 hrs / 2K steps |
| LoRA DPO training (Flux.2) | ~20 GB | ~6-10 hrs / 5K steps |
| Evaluation (FVD/CLIP/VBench) | ~8 GB | ~1 hr / 100 samples |

**Minimum**: Single A100 (80 GB) or 2x RTX 4090 (24 GB each)

**Strategy**: Run generation and scoring sequentially (both need GPU). Training uses single-GPU with gradient accumulation. Evaluation is lightweight.

---

## Timeline Summary

| Week | Phase | Work |
|:-----|:------|:-----|
| 1 | Phase 1 | Generate 500 pairs, score with 3 strategies, collect human annotations |
| 2 | Phase 1→2 | Compute agreement metrics, begin 5K pair generation |
| 3 | Phase 2 | Train main model + baselines, begin ablation grid |
| 4 | Phase 2→3 | Complete ablations, evaluate Exp 3, train image DPO (Exp 2) |
| 5 | Phase 3→4 | Cross-modal transfer (Exp 4), reward decomposition (Exp 5) |
| 6 | Writeup | Consolidate results, figures, paper draft |

---

## Target Venues

- **ML**: ICLR / NeurIPS / ICML
- **Vision**: CVPR / ECCV
- The cross-modal transfer angle (Contribution 2) is particularly novel and differentiating

---

## Key Research Questions

1. **Can VLMs replace humans as preference judges for video?** → Exp 1 (κ > 0.6 threshold)
2. **Does DPO improve video diffusion models measurably?** → Exp 3 (FVD↓, CLIP↑, VBench↑)
3. **Does image→video LoRA transfer work?** → Exp 4 (cheaper alignment path)
4. **Which quality dimension matters most for video DPO?** → Exp 5 (actionable decomposition)
