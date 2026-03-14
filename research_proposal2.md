# Aligning Video Diffusion Models via VLM Preference Optimization

## The Gap

DPO/RLHF has transformed LLMs. For **image** diffusion, Diffusion-DPO exists but relies on **expensive human preference data**. For **video** diffusion, preference-based alignment is **completely unexplored** in open models — no one has done it because collecting human video preferences at scale is impractical.

Meanwhile, VLMs like InternVL-U can now reason about image/video quality, consistency, and prompt adherence with near-human accuracy. **This creates an opportunity: use VLMs as free, scalable preference oracles to align video diffusion models.**

## Core Method

### Diffusion-DPO with VLM Preferences

For a given prompt $p$:

1. Sample **two** videos from the base model: $v_w, v_l$ (winner, loser)
2. **InternVL-U scores both** on multiple axes (prompt adherence, motion quality, temporal consistency, visual quality)
3. The higher-scored video becomes the "preferred" sample
4. Fine-tune the video DiT with the **Diffusion-DPO loss** applied to LoRA parameters:

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(v_w|p)}{\pi_{\text{ref}}(v_w|p)} - \log \frac{\pi_\theta(v_l|p)}{\pi_{\text{ref}}(v_l|p)}\right]\right)$$

where $\pi_\theta$ is the LoRA-tuned model and $\pi_{\text{ref}}$ is the frozen base.

### What's novel here (the ML contributions):

**1. VLM Preference Distillation for Video**
- Design multi-aspect VLM prompting strategies that decompose video quality into scoreable dimensions
- Study: which VLM scoring strategy produces the best training signal? (single holistic score vs. multi-aspect vs. CoT reasoning)
- Contribution: a **preference data generation pipeline** that replaces human annotation

**2. Cross-Modal DPO Transfer (Image → Video)**
- **Key insight:** Start DPO on Flux.2 (T2I, cheaper), then transfer the learned LoRA to the video DiT
- Both Flux.2 and Wan2.1 use DiT architectures — study which LoRA layers transfer and which are modality-specific
- Contribution: a **curriculum strategy** for preference learning across modalities

**3. Multi-Aspect Reward Decomposition**
- Instead of one scalar preference, decompose into: prompt adherence $r_p$, temporal consistency $r_t$, visual quality $r_v$, motion naturalness $r_m$
- Train with a weighted DPO loss: $r = \alpha_p r_p + \alpha_t r_t + \alpha_v r_v + \alpha_m r_m$
- Contribution: study which **reward dimension matters most** for video quality, and whether decomposition beats holistic scoring

## Models Used (all free on HF)

| Model | Role in the project |
|:---|:---|
| **InternVL-U (4B)** | Preference oracle — scores generated content via multi-aspect CoT prompting |
| **Flux.2-dev (32B)** | Image DPO stage — cheaper to train, validates method before scaling to video |
| **Wan2.1-T2V-1.3B** | Primary video model to align — small enough for LoRA DPO on academic GPUs |
| **CogVideoX-5B** | Second video model — verify method generalizes across architectures |
| **Flux.2 / InternVL-U editing** | Generate harder negatives via targeted edits (hard-negative mining) |

## Experiment Plan

### Exp 1: VLM Preference Quality (does InternVL-U agree with humans?)
- Generate 500 video pairs, score with InternVL-U AND human annotators
- Report agreement rate (Cohen's kappa)
- Ablate: holistic score vs. multi-aspect vs. CoT prompting
- **This alone is a contribution** — benchmarking VLMs as video quality judges

### Exp 2: Image DPO on Flux.2 (validate the method works)
- LoRA-DPO on Flux.2 with InternVL-U preferences on 10K image pairs
- Compare to: base Flux.2, human-preference DPO (if data available)
- Metrics: FID, CLIP-score, human preference win-rate

### Exp 3: Video DPO on Wan2.1 (the main result)
- LoRA-DPO on Wan2.1-1.3B with VLM preferences on 5K video pairs
- Baselines: base Wan2.1, SFT-only fine-tuning, random preference (sanity check)
- Metrics: FVD, CLIP-score, VBench, human preference
- Ablate: number of preference pairs, $\beta$ sensitivity, LoRA rank

### Exp 4: Cross-modal transfer (does image DPO help video DPO?)
- Compare: video DPO from scratch vs. initialized from image DPO LoRA
- Study which transformer layers benefit from transfer

### Exp 5: Multi-aspect reward decomposition
- Compare: holistic DPO vs. decomposed multi-aspect DPO
- Ablate reward weights $\alpha_p, \alpha_t, \alpha_v, \alpha_m$
- Find: does temporal consistency reward dominate? Is motion reward redundant with temporal?

## Why This Is Strong Research

| | |
|:---|:---|
| **Novelty** | First DPO for open video diffusion models; first VLM-as-preference-oracle for video alignment |
| **Timing** | DPO + video generation are both peak-interest topics; their intersection is wide open |
| **Method depth** | Three concrete ML contributions (VLM preference distillation, cross-modal transfer, reward decomposition) — any one could be a paper |
| **Clean ablations** | 5 experiments with clear baselines and controlled comparisons |
| **Venue fit** | ICLR / NeurIPS / ICML (main); CVPR/ECCV if positioned as vision |
| **Feasibility** | LoRA training on Wan2.1-1.3B fits on 1-2 GPUs; InternVL-U inference is cheap at 4B |

## Feasibility & Hardware

- **LoRA DPO on Wan2.1-1.3B**: ~16GB VRAM, trains in hours not days
- **InternVL-U preference generation**: ~10GB, can batch offline
- **Flux.2-dev image DPO**: ~20GB with FP8
- **Total**: Doable on a single A100 or 2× RTX 4090
