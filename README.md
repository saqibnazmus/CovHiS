<div align="center">

# CovHiS: Bridging Covariance and Subspace History for Classifier-Free Guidance

**Nazmus Saqib**<sup>1</sup>, **Masud-An Nur Islam Fahim**<sup>2</sup>, **Gyeongmin Kim**<sup>1</sup>, **Joon-Min Gil**<sup>1</sup>

<sup>1</sup>Jeju National University, Republic of Korea &nbsp;&nbsp; <sup>2</sup>University of Vaasa, Finland

**ACCV 2026**

[**Paper**](#) &nbsp;|&nbsp; [**Supplementary**](#) &nbsp;|&nbsp; [**BibTeX**](#citation)

</div>

<p align="center">
  <img src="teaser.png" width="100%" alt="CovHiS teaser"/>
</p>
<p align="center"><em>
CovHiS keeps generations consistent beyond fixed high guidance scales: APG vs. CovHiS for the prompt
"A pink dog" on SDXL, under the same number of function evaluations.
</em></p>

---

## Overview

Classifier-free guidance (CFG) is essential for high-quality conditional generation, but its
scale is a double-edged sword: a large guidance scale ω improves prompt alignment while causing
**over-saturation, contrast burnout and off-manifold drift**, whereas a small ω loses crisp detail.
Existing corrections still multiply the surviving residual by ω, so they only *postpone* the
failure to a larger scale.

**CovHiS** is a **training-free, plug-and-play** CFG correction that works in two stages:

1. **Saturation stabilization (HoRF-CFG).** A short buffer of recent CFG residuals reveals the
   low-rank subspace of guidance directions that are repeatedly reinforced across denoising steps.
   *History-orthogonal residual filtering* attenuates these recurrent components, and an
   *adaptive guidance budget* bounds the effective guidance displacement independently of ω.
2. **Covariance-tangent detail refinement.** The conditional covariance subspace of the
   text-conditioned prediction is reweighted toward prompt-relevant eigen-directions, and only the
   *tangential* part of the resulting detail is injected under a relative norm budget, restoring
   structure without reintroducing saturation.

CovHiS works across diffusion backbones (SD 2.1, SDXL, SD3, DiT-XL/2, SiT-XL+REPA), samplers
(DDIM, DPM++, SDE-DPM++, PNDM, UniPC), score-distilled models (PixArt-δ, SDXL-Lightning), and
text-to-video generation (Mochi), at an inference-time overhead of only **+1.8%**.

---

## Text-to-Video Results (Mochi, ω = 17.5)

<table>
  <tr>
    <td align="center" width="33%">
      <img src="video_suv.gif" width="100%" alt="SUV video"/><br/>
      <sub><em>"The camera follows behind a white vintage SUV with a black roof rack as it speeds
      up a steep dirt road surrounded by pine trees …"</em></sub>
    </td>
    <td align="center" width="33%">
      <img src="video_fish.gif" width="100%" alt="Tropical fish video"/><br/>
      <sub><em>"A vibrant tropical fish glides gracefully through colorful ocean reefs, surrounded
      by swaying coral …"</em></sub>
    </td>
    <td align="center" width="33%">
      <img src="video_man.gif" width="100%" alt="Gray-haired man video"/><br/>
      <sub><em>"An extreme close-up of a gray-haired man with a beard in his 60s, he is deep in
      thought pondering the history of the universe …"</em></sub>
    </td>
  </tr>
  <tr>
    <td align="center"><a href="video_suv.mp4">▶ Full-quality MP4</a></td>
    <td align="center"><a href="video_fish.mp4">▶ Full-quality MP4</a></td>
    <td align="center"><a href="video_man.mp4">▶ Full-quality MP4</a></td>
  </tr>
</table>

---

## Quantitative Results

Results on 30K MS-COCO validation captions at a moderate (ω = 7.5) and a high (ω = 30.5)
guidance scale (Table 1 of the paper).

| Method | SDXL FID ↓ (7.5 / 30.5) | SDXL Saturation ↓ (7.5 / 30.5) | SD3 FID ↓ (7.5 / 30.5) | SD3 Saturation ↓ (7.5 / 30.5) |
|:--|:--:|:--:|:--:|:--:|
| CFG    | 27.83 / 33.55 | 0.28 / 0.46 | 27.19 / 33.32 | 0.34 / 0.44 |
| APG    | 27.35 / 29.08 | 0.18 / 0.23 | 27.02 / 30.45 | 0.32 / 0.36 |
| HiGS   | 27.16 / 29.46 | 0.20 / 0.27 | 26.83 / 31.27 | 0.33 / 0.37 |
| **CovHiS** | **26.94 / 27.95** | **0.11 / 0.14** | **26.51 / 26.56** | **0.27 / 0.28** |

Computational cost on SDXL (NVIDIA RTX 4090):

| Method | vRAM (GB) | Latency (s) | Overhead |
|:--|:--:|:--:|:--:|
| CFG (baseline) | 6.90 | 9.503 | +0.0% |
| **CovHiS** | 6.91 | 9.678 | **+1.8%** |

See the paper and supplementary material for precision, recall, contrast, human-preference
benchmarks (DrawBench, PartiPrompts, HPS), sampler and backbone studies, and ablations.

---

## Installation

```bash
git clone https://github.com/saqibnazmus/CovHiS.git
cd CovHiS
pip install -r requirements.txt
```

## Usage

<!-- TODO: replace this section with the actual entry point of the code. -->

```bash
# Example (update the script name and arguments to match the repository)
python sample.py --model sdxl --prompt "A pink dog" --guidance_scale 30.5
```

### Main hyperparameters

| Symbol | Role | Paper default |
|:--:|:--|:--:|
| ω | CFG guidance scale | — |
| β (−1 < β < 0) | Momentum coefficient of the residual history | — |
| W, r | History-window size and rank of the recurrent subspace | — |
| η (0 ≤ η ≤ 1) | Retention of the history-parallel component | — |
| γ (0 < γ ≤ 1) | Adaptive guidance budget | — |
| β₁ | Structure-refinement (momentum) coefficient | 1 × 10⁻² |
| κ, p | Eigen-direction reweighting coefficients | κ = 1, p = 2 |
| μ | Relative norm budget for tangential detail injection | — |

<!-- TODO: fill in the "—" entries with the defaults used in the code. -->

---

## Citation

If you find CovHiS useful in your research, please cite:

```bibtex
@inproceedings{saqib2026covhis,
  title     = {CovHiS: Bridging Covariance and Subspace History for Classifier-Free Guidance},
  author    = {Saqib, Nazmus and Fahim, Masud-An Nur Islam and Kim, Gyeongmin and Gil, Joon-Min},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Contact

For questions, please open an issue or contact Nazmus Saqib (nsaqib1995@gmail.com).
