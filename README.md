<div align="center">

# CovHiS: Bridging Covariance and Subspace History for Classifier-Free Guidance

**Nazmus Saqib**<sup>1</sup>, **Masud-An Nur Islam Fahim**<sup>2</sup>, **Gyeongmin Kim**<sup>1</sup>, **Joon-Min Gil**<sup>1</sup>

<sup>1</sup>Jeju National University, Republic of Korea &nbsp;&nbsp; <sup>2</sup>University of Vaasa, Finland

### ACCV 2026

[![Paper](https://img.shields.io/badge/Paper-ACCV%202026-blue)](#)
[![Supplementary](https://img.shields.io/badge/Supplementary-PDF-orange)](#)
[![Website](https://img.shields.io/badge/Project-Website-green)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saqibnazmus/CovHiS/blob/main/quick_run.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

<p align="center">
  <img src="teaser.jpg" width="100%" alt="CovHiS teaser"/>
</p>
<p align="center"><em>
CovHiS keeps generations consistent beyond fixed high guidance scales: APG vs. CovHiS for the
prompt "A pink dog" on SDXL, under the same number of function evaluations.
</em></p>

---

## 🔥 News

- **[2026-09-25]** CovHiS has been accepted to **ACCV 2026**! 🎉
- **[2026-10-01]** Code for the SDXL and SD3 pipelines and the quick-run notebook is released.

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

## Text-to-Image Results

<p align="center">
  <img src="T2I.png" width="100%" alt="CovHiS text-to-image results"/>
</p>
<p align="center"><em>
Qualitative comparison of CFG, HiGS, APG and CovHiS on SDXL and SD3 under a high guidance
scale (ω = 30.5).
</em></p>

---


## Quantitative Results

Quantitative comparison with guidance methods on SDXL and SD3.0 (30K MS-COCO validation captions)
under a moderate (ω = 7.5) and a high (ω = 30.5) guidance scale. Each cell reports
**ω = 7.5 / ω = 30.5**; best results are in **bold**.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="5">SDXL</th>
      <th colspan="5">SD3.0</th>
    </tr>
    <tr>
      <th>FID ↓</th><th>Prec. ↑</th><th>Rec. ↑</th><th>Sat. ↓</th><th>Con. ↓</th>
      <th>FID ↓</th><th>Prec. ↑</th><th>Rec. ↑</th><th>Sat. ↓</th><th>Con. ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>CFG</td>
      <td>27.83 / 33.55</td><td>0.61 / 0.49</td><td>0.53 / 0.42</td><td>0.28 / 0.46</td><td>0.23 / 0.37</td>
      <td>27.19 / 33.32</td><td>0.72 / 0.55</td><td>0.41 / 0.22</td><td>0.34 / 0.44</td><td>0.27 / 0.36</td></tr>
    <tr><td>APG</td>
      <td>27.35 / 29.08</td><td>0.64 / <b>0.62</b></td><td>0.54 / 0.52</td><td>0.18 / 0.23</td><td>0.17 / 0.21</td>
      <td>27.02 / 30.45</td><td>0.74 / 0.77</td><td>0.43 / 0.35</td><td>0.32 / 0.36</td><td>0.22 / 0.26</td></tr>
    <tr><td>HiGS</td>
      <td>27.16 / 29.46</td><td>0.65 / 0.59</td><td>0.53 / 0.50</td><td>0.20 / 0.27</td><td>0.18 / 0.23</td>
      <td>26.83 / 31.27</td><td>0.76 / 0.70</td><td>0.42 / 0.33</td><td>0.33 / 0.37</td><td>0.23 / 0.30</td></tr>
    <tr><td>CFG++</td>
      <td>27.24 / –</td><td>0.64 / –</td><td>0.54 / –</td><td>0.22 / –</td><td>0.20 / –</td>
      <td>26.98 / –</td><td>0.76 / –</td><td>0.42 / –</td><td>0.33 / –</td><td>0.24 / –</td></tr>
    <tr><td>TAG</td>
      <td>27.68 / 32.16</td><td>0.62 / 0.51</td><td>0.53 / 0.44</td><td>0.24 / 0.40</td><td>0.20 / 0.32</td>
      <td>27.11 / 32.81</td><td>0.75 / 0.58</td><td>0.41 / 0.24</td><td>0.31 / 0.39</td><td>0.24 / 0.33</td></tr>
    <tr><td>TPG</td>
      <td>27.59 / 31.84</td><td>0.63 / 0.53</td><td>0.55 / 0.46</td><td>0.23 / 0.38</td><td>0.19 / 0.30</td>
      <td>27.93 / 32.76</td><td>0.74 / 0.61</td><td>0.43 / 0.25</td><td>0.32 / 0.40</td><td>0.24 / 0.32</td></tr>
    <tr><td>SAG</td>
      <td>28.12 / 34.02</td><td>0.60 / 0.47</td><td>0.52 / 0.40</td><td>0.29 / 0.47</td><td>0.24 / 0.38</td>
      <td>28.34 / 33.85</td><td>0.70 / 0.59</td><td>0.40 / 0.20</td><td>0.35 / 0.42</td><td>0.26 / 0.36</td></tr>
    <tr><td>PAG</td>
      <td>27.94 / 33.27</td><td>0.62 / 0.50</td><td>0.54 / 0.43</td><td>0.27 / 0.44</td><td>0.22 / 0.35</td>
      <td>28.23 / 33.76</td><td>0.71 / 0.62</td><td>0.40 / 0.21</td><td>0.35 / 0.43</td><td>0.24 / 0.37</td></tr>
    <tr><td>SEG</td>
      <td>27.31 / 31.35</td><td><b>0.66</b> / 0.56</td><td>0.55 / 0.47</td><td>0.21 / 0.34</td><td>0.18 / 0.27</td>
      <td>27.06 / 33.28</td><td>0.78 / 0.68</td><td>0.43 / 0.27</td><td>0.31 / 0.41</td><td>0.23 / 0.29</td></tr>
    <tr><td>ASAG</td>
      <td>27.22 / 30.92</td><td>0.65 / 0.58</td><td>0.56 / 0.49</td><td>0.20 / 0.32</td><td>0.18 / 0.26</td>
      <td>27.01 / 33.14</td><td>0.77 / 0.69</td><td>0.44 / 0.28</td><td>0.30 / 0.42</td><td>0.24 / 0.30</td></tr>
    <tr><td><b>CovHiS</b></td>
      <td><b>26.94 / 27.95</b></td><td>0.65 / 0.60</td><td><b>0.58 / 0.56</b></td><td><b>0.11 / 0.14</b></td><td><b>0.14 / 0.16</b></td>
      <td><b>26.51 / 26.56</b></td><td><b>0.82 / 0.80</b></td><td><b>0.48 / 0.47</b></td><td><b>0.27 / 0.28</b></td><td><b>0.18 / 0.20</b></td></tr>
  </tbody>
</table>

---

## Text-to-Video Quantitative Results

Text-to-video generation with **Mochi** on 100 randomly sampled VBench prompts.

| Method | Dyn. Deg. ↑ | Img. Qual. ↑ | Aes. Qual. ↑ | BG Cons. ↑ | Subj. Cons. ↑ | Overall Cons. ↑ | Human Act. ↑ | Mot. Smooth ↑ | Temp. Style ↑ |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Mochi (CFG) | 0.54 | **0.7132** | 0.5674 | 0.9734 | **0.9675** | 0.1812 | 0.06 | **0.9878** | 0.1782 |
| **+ CovHiS** | **0.57** | 0.7066 | **0.5697** | **0.9783** | 0.9657 | **0.1844** | **0.08** | 0.9868 | **0.1852** |

---

## Clone the Repository

```bash
git clone https://github.com/saqibnazmus/CovHiS.git
cd CovHiS
```

## Create an Environment and Install the Dependencies

```bash
conda create -n covhis python=3.10 -y
conda activate covhis
pip install -r requirements.txt
```

Log in to Hugging Face to download the SDXL and SD3 checkpoints:

```bash
huggingface-cli login
```

> Stable Diffusion 3 is a gated model: accept its license on its Hugging Face model page with the
> same account before running the SD3 pipeline.

---

## Quick Run



```python
# ===== CovHiS quick run: SDXL and SD3 =====
import os, torch
import numpy as np
from matplotlib import pyplot as plt
import gc 
print(os.environ.get("CUDA_VISIBLE_DEVICES"))
print(torch.cuda.device_count())
print(torch.cuda.is_available())


# Option A: just use the first visible GPU
DEVICE = "cuda:0"

# Option B: unhide all GPUs (must run BEFORE importing torch / starting CUDA)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
NUM_STEPS = 28
PROMPT = "A pink dog"
NEGATIVE_PROMPT = ""
SEED = 1
GUIDANCE_SCALE = 30.5

###### run covhis

from pipeline_covhis_sdxl import StableDiffusionXLcovhisPipeline

pipe = StableDiffusionXLcovhisPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to(DEVICE)

generator = torch.Generator(device=DEVICE).manual_seed(SEED)

image = pipe(
    PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=NUM_STEPS,
    guidance_scale=GUIDANCE_SCALE,
    generator=generator,
).images[0]
output = np.concatenate([np.array(image)], 1)

plt.figure(figsize=(16, 8))
plt.imshow(output)

```

---

## Citation

The BibTeX entry will be available after publication.

---

## Contact

For questions, please open an issue or contact:

- Nazmus Saqib: nsaqib1995@gmail.com
- Masud-An Nur Islam Fahim: masud.fahim@uwasa.fi

---

## License

This project is released under the [MIT License](LICENSE).
