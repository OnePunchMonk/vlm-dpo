# VLM-DPO: Aligning Video Diffusion Models via VLM Preference Optimization

**Framework for preference-based alignment of video diffusion models using Vision-Language Models as automated preference oracles.**

## Overview

DPO/RLHF has transformed LLMs, but applying preference optimization to video diffusion is unexplored — human video preference data is impractical to collect at scale. VLM-DPO solves this by using VLMs (InternVL-U) as free, scalable preference judges.

### Key Contributions

1. **VLM Preference Distillation** — Multi-aspect VLM prompting strategies that decompose video quality into scoreable dimensions, replacing human annotation entirely
2. **Cross-Modal DPO Transfer** — Curriculum strategy: train image DPO on Flux.2, then transfer LoRA weights to video DiT (Wan2.1)
3. **Multi-Aspect Reward Decomposition** — Decomposed DPO loss: `r = α_p·r_p + α_t·r_t + α_v·r_v + α_m·r_m` studying which reward dimension matters most

## Models

| Model | Role |
|:---|:---|
| InternVL-U (4B) | Preference oracle — multi-aspect CoT scoring |
| Flux.2-dev (32B) | Image DPO stage (cheaper validation) |
| Wan2.1-T2V-1.3B | Primary video model to align |
| CogVideoX-5B | Generalization verification |

## Installation

```bash
# Clone and install
git clone <repo-url> && cd vlm-dpo
pip install -e ".[all]"
```

## Quick Start

### 1. Generate Preference Data

```bash
# Generate video pairs and score with InternVL-U
vlm-dpo generate --config configs/exp3_video_dpo.yaml
```

### 2. Train with DPO

```bash
# LoRA-DPO on Wan2.1
vlm-dpo train --config configs/exp3_video_dpo.yaml
```

### 3. Evaluate

```bash
# Run metrics (FVD, CLIP-score, VBench)
vlm-dpo evaluate --config configs/exp3_video_dpo.yaml
```

## Experiments

| Exp | Description | Config |
|:---|:---|:---|
| 1 | VLM vs Human agreement (Cohen's κ) | `configs/exp1_vlm_agreement.yaml` |
| 2 | Image DPO on Flux.2 | `configs/exp2_image_dpo.yaml` |
| 3 | Video DPO on Wan2.1 (main result) | `configs/exp3_video_dpo.yaml` |
| 4 | Cross-modal transfer (Image→Video) | `configs/exp4_cross_modal.yaml` |
| 5 | Multi-aspect reward decomposition | `configs/exp5_multi_aspect.yaml` |

## Hardware Requirements

- **LoRA DPO on Wan2.1-1.3B**: ~16GB VRAM
- **InternVL-U preference generation**: ~10GB VRAM
- **Flux.2-dev image DPO**: ~20GB VRAM (FP8)
- **Total**: Single A100 or 2× RTX 4090

## License

Apache 2.0
