# Implementation of StableDiffusionXLcovhisPipeline (CovHiS guidance for SDXL)

import inspect
from typing import Type, Any, Callable, Dict, List, Optional, Tuple, Union
import torch_dct as dct
import math
import copy
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import set_seed

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from pipeline_covhis_sdxl import StableDiffusionXLcovhisPipeline

        >>> pipe = StableDiffusionXLcovhisPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt, guidance_scale=7.5).images[0]
        ```
"""


def dct2(x):
    return dct.dct_2d(x, norm="ortho")


def idct2(x):
    return dct.idct_2d(x, norm="ortho")


class SVDToneTransfer:
    def __init__(
        self,
        tone_strength=0.7,
        mean_strength=0.8,
        max_singular_ratio=1.5,
        eps=1e-8,
    ):
        self.tone_strength = tone_strength
        self.mean_strength = mean_strength
        self.max_singular_ratio = max_singular_ratio
        self.eps = eps

    def update(self, noise_pred_cov, noise_pred_aux):
        if noise_pred_cov.shape != noise_pred_aux.shape:
            raise ValueError(
                f"Shape mismatch: {noise_pred_cov.shape} != "
                f"{noise_pred_aux.shape}"
            )

        if noise_pred_cov.ndim < 3:
            raise ValueError(
                f"Expected [B,C,...], received {noise_pred_cov.shape}"
            )

        original_dtype = noise_pred_cov.dtype

        cov = noise_pred_cov.float()
        aux = noise_pred_aux.float()

        batch_size, channels = cov.shape[:2]

        cov_matrix = cov.reshape(batch_size, channels, -1)
        aux_matrix = aux.reshape(batch_size, channels, -1)

        cov_mean = cov_matrix.mean(dim=-1, keepdim=True)
        aux_mean = aux_matrix.mean(dim=-1, keepdim=True)

        cov_centered = cov_matrix - cov_mean
        aux_centered = aux_matrix - aux_mean

        U_cov, S_cov, Vh_cov = torch.linalg.svd(
            cov_centered,
            full_matrices=False,
        )

        _, S_aux, _ = torch.linalg.svd(
            aux_centered,
            full_matrices=False,
        )

        singular_ratio = S_aux / (S_cov + self.eps)

        singular_ratio = singular_ratio.clamp(
            min=1.0 / self.max_singular_ratio,
            max=self.max_singular_ratio,
        )

        S_target = S_cov * singular_ratio

        S_new = (
            S_cov
            + self.tone_strength
            * (S_target - S_cov)
        )

        mean_new = (
            cov_mean
            + self.mean_strength
            * (aux_mean - cov_mean)
        )

        transferred = torch.matmul(
            U_cov * S_new.unsqueeze(-2),
            Vh_cov,
        )

        transferred = transferred + mean_new

        return transferred.reshape_as(cov).to(original_dtype)


class TextCovarianceTangentGuidance:
    def __init__(
        self,
        covariance_gain=2.0,
        eigen_power=1.5,
        strength=1.5,
        max_ratio=0.25,
        eps=1e-8,
    ):
        self.covariance_gain = covariance_gain
        self.eigen_power = eigen_power
        self.strength = strength
        self.max_ratio = max_ratio
        self.eps = eps
        self.last_correction_ratio = 0.0

    def update(
        self,
        noise_pred_base,
        noise_pred_uncond,
        noise_pred_text,
    ):
        if (
            noise_pred_base.shape != noise_pred_uncond.shape
            or noise_pred_base.shape != noise_pred_text.shape
        ):
            raise ValueError(
                f"Shape mismatch: base={noise_pred_base.shape}, "
                f"uncond={noise_pred_uncond.shape}, "
                f"text={noise_pred_text.shape}"
            )

        if noise_pred_base.ndim < 3:
            raise ValueError(
                f"Expected [B,C,...], received {noise_pred_base.shape}"
            )

        original_dtype = noise_pred_base.dtype
        base = noise_pred_base.float()
        uncond = noise_pred_uncond.float()
        text = noise_pred_text.float()

        batch_size, channels = base.shape[:2]

        base_flat = base.reshape(batch_size, channels, -1)
        uncond_flat = uncond.reshape(batch_size, channels, -1)
        text_flat = text.reshape(batch_size, channels, -1)

        guidance_flat = text_flat - uncond_flat

        text_centered = text_flat - text_flat.mean(
            dim=-1,
            keepdim=True,
        )

        spatial_size = text_centered.shape[-1]

        covariance = torch.matmul(
            text_centered,
            text_centered.transpose(-1, -2),
        ) / max(spatial_size - 1, 1)

        covariance = covariance + self.eps * torch.eye(
            channels,
            device=covariance.device,
            dtype=covariance.dtype,
        ).unsqueeze(0)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

        eigenvalues = eigenvalues.clamp_min(0.0)

        normalized_eigenvalues = eigenvalues / (
            eigenvalues.amax(dim=-1, keepdim=True) + self.eps
        )

        metric_weights = 1.0 + self.covariance_gain * (
            normalized_eigenvalues.pow(self.eigen_power)
        )

        guidance_coordinates = torch.matmul(
            eigenvectors.transpose(-1, -2),
            guidance_flat,
        )

        weighted_guidance_coordinates = (
            metric_weights.unsqueeze(-1)
            * guidance_coordinates
        )

        weighted_guidance = torch.matmul(
            eigenvectors,
            weighted_guidance_coordinates,
        )

        covariance_correction = (
            weighted_guidance - guidance_flat
        )

        correction_vector = covariance_correction.reshape(
            batch_size,
            -1,
        )

        uncond_vector = uncond.reshape(batch_size, -1)

        parallel_coefficient = (
            correction_vector * uncond_vector
        ).sum(dim=1, keepdim=True) / (
            uncond_vector.square().sum(
                dim=1,
                keepdim=True,
            )
            + self.eps
        )

        radial_correction = (
            parallel_coefficient * uncond_vector
        )

        tangent_correction = (
            correction_vector - radial_correction
        )

        proposed_correction = (
            self.strength * tangent_correction
        )

        base_vector = base.reshape(batch_size, -1)

        base_norm = torch.linalg.vector_norm(
            base_vector,
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        correction_norm = torch.linalg.vector_norm(
            proposed_correction,
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        trust_scale = torch.clamp(
            self.max_ratio * base_norm / correction_norm,
            max=1.0,
        )

        final_correction = (
            trust_scale * proposed_correction
        )

        refined = (
            base_vector + final_correction
        ).reshape_as(base)

        self.last_correction_ratio = (
            torch.linalg.vector_norm(
                final_correction,
                dim=1,
            )
            / torch.linalg.vector_norm(
                base_vector,
                dim=1,
            ).clamp_min(self.eps)
        ).mean().item()

        return refined.to(original_dtype)


class PersistentDetailSubspace:
    def __init__(
        self,
        window=4,
        rank=2,
        ema_alpha=0.7,
        strength=0.15,
        max_ratio=0.05,
        eps=1e-8,
    ):
        self.history = deque(maxlen=window)
        self.rank = rank
        self.ema_alpha = ema_alpha
        self.strength = strength
        self.max_ratio = max_ratio
        self.eps = eps
        self.ema = None

    def reset(self):
        self.history.clear()
        self.ema = None

    def update(self, noise_pred):
        if noise_pred.ndim < 2:
            raise ValueError(
                f"Expected shape [B, ...], received {noise_pred.shape}"
            )

        original_dtype = noise_pred.dtype
        current = noise_pred.float()
        batch_size = current.shape[0]
        broadcast_shape = [batch_size] + [1] * (current.ndim - 1)

        if self.ema is None:
            self.ema = current.detach().clone()
            return noise_pred

        if self.ema.shape != current.shape:
            self.reset()
            self.ema = current.detach().clone()
            return noise_pred

        residual = current - self.ema

        self.ema = (
            self.ema_alpha * current.detach()
            + (1.0 - self.ema_alpha) * self.ema
        )

        self.history.append(residual.detach())

        if len(self.history) < 2:
            return noise_pred

        residual_flat = residual.reshape(batch_size, -1)

        history_matrix = torch.stack(
            [
                item.reshape(batch_size, -1)
                for item in self.history
            ],
            dim=1,
        )

        _, _, Vh = torch.linalg.svd(
            history_matrix,
            full_matrices=False,
        )

        current_rank = min(
            self.rank,
            Vh.shape[1],
            Vh.shape[2],
        )

        if current_rank < 1:
            return noise_pred

        basis = Vh[:, :current_rank, :]

        coefficients = torch.einsum(
            "bd,brd->br",
            residual_flat,
            basis,
        )

        persistent_detail_flat = torch.einsum(
            "br,brd->bd",
            coefficients,
            basis,
        )

        persistent_detail = persistent_detail_flat.reshape_as(
            current
        )

        base_norm = torch.linalg.vector_norm(
            current.reshape(batch_size, -1),
            dim=1,
        ).reshape(broadcast_shape).clamp_min(self.eps)

        detail_norm = torch.linalg.vector_norm(
            persistent_detail.reshape(batch_size, -1),
            dim=1,
        ).reshape(broadcast_shape).clamp_min(self.eps)

        scale = torch.clamp(
            self.max_ratio * base_norm / detail_norm,
            max=1.0,
        )

        refined = (
            current
            + self.strength
            * scale
            * persistent_detail
        )

        return refined.to(original_dtype)


def compute_tcfg_noise_pred(
    noise_pred_uncond: torch.Tensor,
    noise_pred_text: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Compute Tangential Damping Classifier-Free Guidance (TCFG).

    Parameters
    ----------
    noise_pred_uncond:
        Unconditional noise prediction with shape [B, C, H, W].

    noise_pred_text:
        Text-conditional noise prediction with the same shape.

    guidance_scale:
        Classifier-free guidance scale.

    Returns
    -------
    noise_pred:
        Final TCFG-guided noise prediction with shape [B, C, H, W].
    """

    if noise_pred_uncond.shape != noise_pred_text.shape:
        raise ValueError(
            "`noise_pred_uncond` and `noise_pred_text` must have identical shapes, "
            f"but received {noise_pred_uncond.shape} and {noise_pred_text.shape}."
        )

    if noise_pred_uncond.ndim < 2:
        raise ValueError(
            "The prediction tensors must contain a batch dimension and at least "
            "one feature dimension."
        )

    original_dtype = noise_pred_uncond.dtype
    original_shape = noise_pred_uncond.shape
    batch_size = original_shape[0]

    # Create the joint score matrix:
    #
    #     A = [epsilon_text, epsilon_uncond]
    #
    # Shape after flattening: [B, 2, C*H*W].
    all_noise = torch.stack(
        [noise_pred_text, noise_pred_uncond],
        dim=1,
    ).float()

    all_noise = all_noise.reshape(batch_size, 2, -1)

    # Compute the joint singular directions.
    #
    # Vh[:, 0] is the dominant right singular vector and is treated
    # as the shared normal direction.
    _, _, Vh = torch.linalg.svd(
        all_noise,
        full_matrices=False,
    )

    # Keep only the first/dominant singular direction.
    #
    # The second singular vector is interpreted as a less-aligned
    # tangential component and is removed.
    Vh_modified = Vh.clone()
    Vh_modified[:, 1:, :] = 0.0

    # Flatten the unconditional prediction.
    noise_uncond_flat = noise_pred_uncond.float().reshape(
        batch_size,
        1,
        -1,
    )

    # Obtain the unconditional coordinates in the joint SVD basis:
    #
    #     coordinates = epsilon_uncond V^T
    uncond_coordinates = torch.matmul(
        noise_uncond_flat,
        Vh.transpose(-2, -1),
    )

    # Reconstruct using only the dominant singular direction:
    #
    #     epsilon_uncond_hat
    #         = epsilon_uncond V^T [v1, 0]
    modified_uncond_flat = torch.matmul(
        uncond_coordinates,
        Vh_modified,
    )

    modified_uncond = modified_uncond_flat.reshape(
        original_shape
    ).to(
        device=noise_pred_uncond.device,
        dtype=original_dtype,
    )

    # Apply CFG using the modified unconditional prediction:
    #
    # epsilon_TCFG
    #     = epsilon_uncond_hat
    #       + w(epsilon_text - epsilon_uncond_hat)
    noise_pred = modified_uncond + guidance_scale * (
        noise_pred_text - modified_uncond
    )

    return noise_pred


class TangentSubspaceTrustRegion:
    def __init__(self, window=5, rank=3, strength=0.5, max_ratio=0.05, eps=1e-8):
        self.history = deque(maxlen=window)
        self.rank = rank
        self.strength = strength
        self.max_ratio = max_ratio
        self.eps = eps
        self.previous = None

    def reset(self):
        self.history.clear()
        self.previous = None

    def update(self, noise_pred_base, candidate_detail):
        if noise_pred_base.shape != candidate_detail.shape:
            raise ValueError("noise_pred_base and candidate_detail must have identical shapes")

        dtype = noise_pred_base.dtype
        base = noise_pred_base.float()
        candidate = candidate_detail.float()
        batch_size = base.shape[0]

        if self.previous is None or self.previous.shape != base.shape:
            self.previous = base.detach().clone()
            return noise_pred_base

        increment = base - self.previous
        self.previous = base.detach().clone()
        self.history.append(increment.detach())

        if len(self.history) < 2:
            return noise_pred_base

        history_matrix = torch.stack(
            [item.reshape(batch_size, -1) for item in self.history],
            dim=1,
        )

        _, _, Vh = torch.linalg.svd(history_matrix, full_matrices=False)
        current_rank = min(self.rank, Vh.shape[1])
        basis = Vh[:, :current_rank, :]

        candidate_flat = candidate.reshape(batch_size, -1)
        coefficients = torch.einsum("bd,brd->br", candidate_flat, basis)
        tangent_flat = torch.einsum("br,brd->bd", coefficients, basis)

        base_flat = base.reshape(batch_size, -1)
        parallel_scale = (tangent_flat * base_flat).sum(dim=1, keepdim=True) / (
            base_flat.square().sum(dim=1, keepdim=True) + self.eps
        )

        safe_flat = tangent_flat - parallel_scale * base_flat

        base_norm = torch.linalg.vector_norm(base_flat, dim=1, keepdim=True)
        safe_norm = torch.linalg.vector_norm(safe_flat, dim=1, keepdim=True)

        trust_scale = torch.clamp(
            self.max_ratio * base_norm / (safe_norm + self.eps),
            max=1.0,
        )

        refined_flat = base_flat + self.strength * trust_scale * safe_flat
        return refined_flat.reshape_as(base).to(dtype)


def compute_g_brkt(
    noise_pred_uncond: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    rank_each: int = 2,
    tau: float = 0.96,
    conflict_damping: float = 0.25,
    eps: float = 1e-8,
    max_norm_gain: float | None = None,
) -> torch.Tensor:
    """
    Compute the BRKT-refined CFG residual.

    Inputs
    ------
    noise_pred_uncond:
        Unconditional noise prediction, shape [B, C, H, W].

    noise_pred_cond:
        Conditional noise prediction, shape [B, C, H, W].

    rank_each:
        Number of dominant singular modes retained from each branch.

    tau:
        Statistical threshold for detecting unusually large off-diagonal
        interactions. Lower tau modifies more coefficients.

    conflict_damping:
        Fraction of each detected interaction retained.
        For example, 0.25 keeps 25% of the detected coefficient.

    eps:
        Numerical-stability constant.

    max_norm_gain:
        Optional upper bound for norm restoration.
        None reproduces the original BRKT operation exactly.

    Returns
    -------
    g_brkt:
        Refined CFG residual with shape [B, C, H, W].
    """

    if noise_pred_uncond.shape != noise_pred_cond.shape:
        raise ValueError(
            "`noise_pred_uncond` and `noise_pred_cond` must have identical shapes, "
            f"but received {noise_pred_uncond.shape} and {noise_pred_cond.shape}."
        )

    if noise_pred_uncond.ndim != 4:
        raise ValueError(
            "BRKT expects predictions with shape [batch, channels, height, width], "
            f"but received a {noise_pred_uncond.ndim}D tensor."
        )

    if rank_each < 1:
        raise ValueError("`rank_each` must be at least 1.")

    if tau < 0:
        raise ValueError("`tau` must be non-negative.")

    if not 0.0 <= conflict_damping <= 1.0:
        raise ValueError("`conflict_damping` must be in [0, 1].")

    original_dtype = noise_pred_uncond.dtype
    batch_size, channels, height, width = noise_pred_uncond.shape

    # Raw CFG residual:
    # G = epsilon_cond - epsilon_uncond
    guidance = noise_pred_cond - noise_pred_uncond
    g_brkt = torch.empty_like(guidance)

    for b in range(batch_size):
        # Convert each prediction into a channel-by-space matrix.
        # Shape: [C, H*W]
        uncond_matrix = noise_pred_uncond[b].float().reshape(channels, -1)
        cond_matrix = noise_pred_cond[b].float().reshape(channels, -1)
        guidance_matrix = cond_matrix - uncond_matrix

        # Estimate dominant channel and spatial modes separately for
        # the unconditional and conditional predictions.
        U_uncond, _, Vh_uncond = torch.linalg.svd(uncond_matrix, full_matrices=False)
        U_cond, _, Vh_cond = torch.linalg.svd(cond_matrix, full_matrices=False)

        # Select the strongest valid modes from each branch.
        current_rank = min(
            rank_each,
            U_uncond.shape[1],
            U_cond.shape[1],
            Vh_uncond.shape[0],
            Vh_cond.shape[0],
        )

        # Combine conditional and unconditional channel modes.
        # Shape: [C, 2*current_rank]
        U_joint = torch.cat(
            [
                U_uncond[:, :current_rank],
                U_cond[:, :current_rank],
            ],
            dim=1,
        )

        # Combine conditional and unconditional spatial modes.
        # Shape: [H*W, 2*current_rank]
        V_joint = torch.cat(
            [
                Vh_uncond[:current_rank].transpose(0, 1),
                Vh_cond[:current_rank].transpose(0, 1),
            ],
            dim=1,
        )

        # Estimate normalized shared bases.
        # Singular values are discarded because the shared bases represent
        # orientation rather than the original branch magnitudes.
        P_u, _, Qh_u = torch.linalg.svd(U_joint, full_matrices=False)
        P_v, _, Qh_v = torch.linalg.svd(V_joint, full_matrices=False)

        U_shared = P_u @ Qh_u
        V_shared = P_v @ Qh_v

        # Express the CFG residual inside the joint channel-spatial basis.
        #
        # A[i, j] measures the interaction between shared channel mode i
        # and shared spatial mode j.
        guidance_coordinates = U_shared.transpose(0, 1) @ guidance_matrix @ V_shared

        # Protect diagonal mode interactions and inspect only off-diagonal
        # cross-mode interactions.
        diagonal_mask = torch.eye(
            guidance_coordinates.shape[0],
            guidance_coordinates.shape[1],
            device=guidance_coordinates.device,
            dtype=torch.bool,
        )

        off_diagonal_values = guidance_coordinates[~diagonal_mask]
        refined_coordinates = guidance_coordinates.clone()

        if off_diagonal_values.numel() > 1:
            # Estimate the center and spread of cross-mode interactions.
            off_mean = off_diagonal_values.mean()
            off_std = off_diagonal_values.std(unbiased=False).clamp_min(eps)

            # Detect unusual off-diagonal interactions.
            conflict_mask = (
                (~diagonal_mask)
                & ((guidance_coordinates - off_mean).abs() >= tau * off_std)
            )

            # Suppress only the detected interactions.
            refined_coordinates[conflict_mask] *= conflict_damping

        # Reconstruct the component represented by the shared basis.
        guidance_inside = U_shared @ guidance_coordinates @ V_shared.transpose(0, 1)

        # Preserve the component outside the selected shared basis.
        guidance_outside = guidance_matrix - guidance_inside

        # Reconstruct the shared-basis component after conflict damping.
        guidance_inside_refined = (
            U_shared
            @ refined_coordinates
            @ V_shared.transpose(0, 1)
        )

        # Combine the preserved outside component with the refined inside part.
        refined_matrix = guidance_outside + guidance_inside_refined

        # Restore the original guidance norm.
        original_norm = guidance_matrix.norm().clamp_min(eps)
        refined_norm = refined_matrix.norm().clamp_min(eps)
        norm_gain = original_norm / refined_norm

        # Optional protection against excessive amplification.
        # max_norm_gain=None reproduces the original BRKT code.
        if max_norm_gain is not None:
            norm_gain = norm_gain.clamp(max=max_norm_gain)

        refined_matrix = refined_matrix * norm_gain

        # Restore [C, H, W] and the original model dtype.
        g_brkt[b] = refined_matrix.reshape(channels, height, width).to(original_dtype)

    return g_brkt


class EMAK:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None  # Don't assume shape, initialize on first call

    def __call__(self, x):
        if self.value is None:
            # First call: initialize with x * alpha
            self.value = self.alpha * x
        else:
            # Subsequent calls: update EMA
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class AdamTracker:
    def __init__(self, lr=0.05, betas=(0.9, 0.99), eps=1e-6, clamp_grad=10.0):
        self.lr = lr
        self.b1, self.b2 = betas          # ← lower β₂ is the key!
        self.eps = eps
        self.clamp_grad = clamp_grad      # ← hard clamp the pseudo-gradient
        self.m = None
        self.v = None
        self.t = 0

    def update(self, V, Ve):
        self.t += 1
        g = V - Ve

        # CRITICAL FIXES:
        g = torch.clamp(g, -self.clamp_grad, self.clamp_grad)   # ← clamp
        if self.m is None:
            self.m = torch.zeros_like(g)
            self.v = torch.zeros_like(g)

        # Slightly lower β₂ + higher eps = rock-solid early steps
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)

        # Bias correction (safe even at t=1)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)

        # Extra safety: clamp v_hat to prevent sqrt(0) or explosion
        step = self.lr * m_hat / (torch.sqrt(v_hat).clamp_min_(1e-6) + self.eps)

        V_new = V - step
        return V_new


class TweedieConsensusFusion:
    def __init__(
        self,
        detail_strength=0.8,
        tone_strength=0.10,
        uncertainty_scale=2.0,
        max_ratio=0.25,
        eps=1e-8,
    ):
        self.detail_strength = detail_strength
        self.tone_strength = tone_strength
        self.uncertainty_scale = uncertainty_scale
        self.max_ratio = max_ratio
        self.eps = eps
        self.last_confidence = 0.0
        self.last_correction_ratio = 0.0

    def _alpha_sigma(self, timestep, scheduler, device):
        if not hasattr(scheduler, "alphas_cumprod"):
            raise ValueError(
                "This version requires a scheduler with alphas_cumprod."
            )

        if torch.is_tensor(timestep):
            timestep = int(timestep.detach().flatten()[0].item())
        else:
            timestep = int(timestep)

        alpha_bar = scheduler.alphas_cumprod[timestep].to(
            device=device,
            dtype=torch.float32,
        )

        alpha = alpha_bar.sqrt().clamp_min(self.eps)
        sigma = (1.0 - alpha_bar).clamp_min(0.0).sqrt().clamp_min(self.eps)

        return alpha, sigma

    def _prediction_type(self, scheduler):
        prediction_type = getattr(
            scheduler.config,
            "prediction_type",
            "epsilon",
        )

        if prediction_type not in {
            "epsilon",
            "v_prediction",
            "sample",
        }:
            raise ValueError(
                f"Unsupported prediction type: {prediction_type}"
            )

        return prediction_type

    def _to_x0(
        self,
        latents,
        model_output,
        alpha,
        sigma,
        prediction_type,
    ):
        if prediction_type == "epsilon":
            return (latents - sigma * model_output) / alpha

        if prediction_type == "v_prediction":
            return alpha * latents - sigma * model_output

        return model_output

    def _from_x0(
        self,
        latents,
        x0,
        alpha,
        sigma,
        prediction_type,
    ):
        if prediction_type == "epsilon":
            return (latents - alpha * x0) / sigma

        if prediction_type == "v_prediction":
            return (alpha * latents - x0) / sigma

        return x0

    def update(
        self,
        latents,
        noise_pred_base,
        noise_pred_cov,
        timestep,
        scheduler,
    ):
        if latents.shape != noise_pred_base.shape:
            raise ValueError(
                f"Latent/base mismatch: "
                f"{latents.shape} != {noise_pred_base.shape}"
            )

        if noise_pred_base.shape != noise_pred_cov.shape:
            raise ValueError(
                f"Base/covariance mismatch: "
                f"{noise_pred_base.shape} != {noise_pred_cov.shape}"
            )

        if latents.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], received {latents.shape}"
            )

        original_dtype = noise_pred_base.dtype

        x_t = latents.float()
        pred_base = noise_pred_base.float()
        pred_cov = noise_pred_cov.float()

        alpha, sigma = self._alpha_sigma(
            timestep,
            scheduler,
            x_t.device,
        )

        prediction_type = self._prediction_type(scheduler)

        x0_base = self._to_x0(
            x_t,
            pred_base,
            alpha,
            sigma,
            prediction_type,
        )

        x0_cov = self._to_x0(
            x_t,
            pred_cov,
            alpha,
            sigma,
            prediction_type,
        )

        innovation = x0_cov - x0_base

        tone = innovation.mean(
            dim=(-2, -1),
            keepdim=True,
        )

        detail = innovation - tone

        base_scale = x0_base.std(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False,
        ).clamp_min(self.eps)

        disagreement = detail.std(
            dim=(-2, -1),
            keepdim=True,
            unbiased=False,
        ) / base_scale

        confidence = 1.0 / (
            1.0
            + self.uncertainty_scale
            * disagreement.square()
        )

        correction = confidence * (
            self.detail_strength * detail
            + self.tone_strength * tone
        )

        batch_size = x0_base.shape[0]

        base_flat = x0_base.reshape(batch_size, -1)
        correction_flat = correction.reshape(batch_size, -1)

        base_norm = torch.linalg.vector_norm(
            base_flat,
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        correction_norm = torch.linalg.vector_norm(
            correction_flat,
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        trust_scale = torch.clamp(
            self.max_ratio * base_norm / correction_norm,
            max=1.0,
        )

        correction_flat = correction_flat * trust_scale
        correction = correction_flat.reshape_as(x0_base)

        x0_new = x0_base + correction

        noise_pred_new = self._from_x0(
            x_t,
            x0_new,
            alpha,
            sigma,
            prediction_type,
        )

        self.last_confidence = confidence.mean().item()

        self.last_correction_ratio = (
            torch.linalg.vector_norm(
                correction_flat,
                dim=1,
            )
            / base_norm.squeeze(1)
        ).mean().item()

        return noise_pred_new.to(original_dtype)


def subspace_distribution_transfer(
    source: torch.Tensor,
    reference: torch.Tensor,
    rank: int = 2,
    strength: float = 0.4,
    mean_strength: float = 0.2,
    max_scale: float = 1.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Transfer dominant channel-subspace statistics from reference to source.

    source, reference: [B, C, H, W]
    """

    if source.shape != reference.shape:
        raise ValueError("source and reference must have identical shapes")

    dtype = source.dtype
    batch, channels, height, width = source.shape
    output = torch.empty_like(source)

    for b in range(batch):
        x = source[b].float().reshape(channels, -1)
        y = reference[b].float().reshape(channels, -1)

        mean_x = x.mean(dim=1, keepdim=True)
        mean_y = y.mean(dim=1, keepdim=True)

        x_centered = x - mean_x
        y_centered = y - mean_y

        covariance_y = (
            y_centered @ y_centered.transpose(0, 1)
        ) / max(y_centered.shape[1] - 1, 1)

        covariance_y = covariance_y + eps * torch.eye(
            channels,
            device=x.device,
            dtype=x.dtype,
        )

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_y)
        order = torch.argsort(eigenvalues, descending=True)

        actual_rank = min(rank, channels)
        basis = eigenvectors[:, order[:actual_rank]]

        source_coordinates = basis.transpose(0, 1) @ x_centered
        reference_coordinates = basis.transpose(0, 1) @ y_centered

        source_std = source_coordinates.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(eps)

        reference_std = reference_coordinates.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(eps)

        scale = (reference_std / source_std).clamp(
            min=1.0 / max_scale,
            max=max_scale,
        )

        target_coordinates = source_coordinates * scale

        subspace_correction = basis @ (
            target_coordinates - source_coordinates
        )

        transferred_mean = mean_x + mean_strength * (
            mean_y - mean_x
        )

        result = (
            x_centered
            + strength * subspace_correction
            + transferred_mean
        )

        output[b] = result.reshape(
            channels,
            height,
            width,
        ).to(dtype)

    return output


def simple_distribution_transfer(source, reference):
    """
    Transfers the distribution (mean and std) of reference to source tensor.
    Works on [B, C, H, W] float tensors, per batch and channel, keeping specified channels unchanged.

    Args:
        source: torch.Tensor [B, C, H, W]
        reference: torch.Tensor [B, C, H, W]

    Returns:
        matched: torch.Tensor [B, C, H, W] with reference distribution applied, except for channels 0 and 1
    """
    assert source.shape == reference.shape, "Tensors must have same shape"

    # Deep copy source to preserve original channels
    matched = copy.deepcopy(source)

    # Compute reference statistics (mean and std over H, W dimensions)
    ref_mean = reference.mean(dim=(2, 3), keepdim=True)
    ref_std = reference.std(dim=(2, 3), keepdim=True) + 1e-8

    # Process each batch
    for b in range(source.shape[0]):
        # Skip channels 0 and 1
        for c in range(2, source.shape[1]):  # Start from channel 2
            src = source[b, c]

            # Source statistics
            src_mean = src.mean()
            src_std = src.std() + 1e-8

            # Standardize and apply reference statistics
            matched[b, c] = (src - src_mean) * (ref_std[b, c] / src_std) + ref_mean[b, c]

    return matched


def compute_gradient(source, reference):
    """
    Computes the gradient of source tensor wrt a loss defined by reference tensor.
    Works on [B, C, H, W] float tensors.

    Args:
        source: torch.Tensor [B, C, H, W], requires_grad=True for gradient computation
        reference: torch.Tensor [B, C, H, W]

    Returns:
        grad_source: torch.Tensor [B, C, H, W], gradient of source wrt MSE loss with reference
    """
    assert source.shape == reference.shape, "Tensors must have same shape"

    # Ensure source requires gradient
    if not source.requires_grad:
        source = source.requires_grad_(True)

    # Define loss: MSE between source and reference
    loss = torch.mean((source - reference) ** 2)

    # Compute gradient of loss with respect to source
    loss.backward()
    grad_source = source.grad.clone()

    return grad_source


def apply_high_freq_dct_mask(diff, threshold=0.05, sharpness=50):
    B, C, H, W = diff.shape
    device = diff.device

    X = dct2(diff)

    u = torch.arange(H, device=device).view(H, 1) / H
    v = torch.arange(W, device=device).view(1, W) / W
    d = torch.sqrt(u**2 + v**2)  # normalized distance from top-left (DC)

    mask = torch.sigmoid((d - threshold) * sharpness)
    X_filtered = X * mask  # broadcast over (B, C)

    diff_filtered = idct2(X_filtered).to(diff.dtype)

    return diff_filtered


def gaussian_smoothing(tensor, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian smoothing to a tensor with shape [B, C, H, W], preserving dimensions.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W]
        kernel_size (int): Size of the Gaussian kernel (must be odd)
        sigma (float): Standard deviation for Gaussian kernel

    Returns:
        torch.Tensor: Smoothed tensor with same shape as input
    """
    # Store original dtype and device
    original_dtype = tensor.dtype
    device = tensor.device

    # Convert to float32 if not already
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)

    # Get input dimensions
    batch_size, channels, height, width = tensor.shape

    # Create Gaussian kernel
    def get_gaussian_kernel(kernel_size, sigma, device):
        x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
        gaussian = torch.exp(-(x**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        kernel = gaussian[:, None] * gaussian[None, :]
        kernel = kernel / kernel.sum()  # Normalize
        return kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

    # Create convolution layer
    kernel = get_gaussian_kernel(kernel_size, sigma, device)
    conv = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,  # Ensure same output size
        groups=channels,  # Apply separately to each channel
        bias=False,
        device=device
    )

    # Set kernel weights and make non-trainable
    conv.weight.data = kernel
    conv.weight.requires_grad = False

    # Apply smoothing
    smoothed_tensor = conv(tensor)

    # Verify output dimensions match input
    assert smoothed_tensor.shape == tensor.shape, \
        f"Output shape {smoothed_tensor.shape} does not match input shape {tensor.shape}"

    # Convert back to original dtype
    if original_dtype != torch.float32:
        smoothed_tensor = smoothed_tensor.to(original_dtype)

    return smoothed_tensor


def enhance_gradient_svd(
    tensor: torch.Tensor,
    latent_fraction: float = 0.8,
    alpha: float = 1.0,
    beta: float = 1.2,
    zero_fraction: float = 0.3,
    calc_similarity: bool = False
) -> torch.Tensor:

    up = tensor.dtype
    tensor = tensor.to(torch.float32)

    if tensor.numel() == 0:
        return tensor

    # Validate the fraction inputs
    if not 0.0 <= latent_fraction <= 1.0:
        raise ValueError("latent_fraction must be between 0.0 and 1.0.")
    if not 0.0 <= zero_fraction <= 1.0:
        raise ValueError("zero_fraction must be between 0.0 and 1.0.")

    original_shape = tensor.shape
    original_tensor_flat = tensor.flatten()

    # Handle 1D tensors by treating them as a row vector (1, N)
    is_1d = (tensor.dim() == 1)
    if is_1d:
        tensor = tensor.unsqueeze(0)

    # torch.linalg.svd works on the last two dimensions
    u, s, vh = torch.linalg.svd(tensor, full_matrices=False)

    # Calculate the target latent size from the fraction
    total_singular_values = s.shape[-1]
    target_latent_size = int(total_singular_values * latent_fraction)

    # Ensure the calculated size is valid
    actual_latent_size = max(0, min(target_latent_size, total_singular_values))

    # If actual_latent_size is 0, the result is a zero tensor
    if actual_latent_size == 0:
        reconstructed_tensor = torch.zeros_like(tensor)
    else:
        # Truncate U, S, and Vh to the target latent size
        u_k = u[..., :actual_latent_size]
        s_k = s[..., :actual_latent_size]
        vh_k = vh[..., :actual_latent_size, :]

        # Apply the exponential decay and scaling to the singular values
        s_modified = s_k * torch.exp(-alpha * s_k) * beta

        # Optionally, zero out a fraction of the smallest singular values
        if 0 < zero_fraction <= 1.0:
            zero_idx = int(actual_latent_size * (1.0 - zero_fraction))
            if zero_idx < actual_latent_size:
                s_modified[..., zero_idx:] = 0.0

        # Reconstruct the tensor from the modified components
        reconstructed_tensor = torch.matmul(u_k * s_modified.unsqueeze(-2), vh_k)

    # If the original tensor was 1D, remove the extra dimension
    if is_1d:
        reconstructed_tensor = reconstructed_tensor.squeeze(0)

    # Calculate and print similarity if requested
    if calc_similarity:
        # Use cosine similarity on the flattened tensors for a general measure
        similarity = F.cosine_similarity(original_tensor_flat, reconstructed_tensor.flatten(), dim=0)
        print(f'Cosine similarity between original and enhanced gradient: {similarity.item():.6f}')

    return reconstructed_tensor.to(up)


def optimized_scale(noise_pred_text, noise_pred_uncond, batch_size):

    positive_flat = noise_pred_text.view(batch_size, -1)
    negative_flat = noise_pred_uncond.view(batch_size, -1)

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    alpha = dot_product / squared_norm

    alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))

    alpha = alpha.to(positive_flat.dtype)

    return alpha


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = None  # Lazy initialization

    def update(self, update_value: torch.Tensor):
        if self.running_average is None:
            self.running_average = torch.zeros_like(update_value, device=update_value.device)
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(v0: torch.Tensor, v1: torch.Tensor):
    dtype = v0.dtype
    device = v0.device
    v0, v1 = v0.double(), v1.double().to(device)
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device), v0_orthogonal.to(dtype).to(device)


def adaptive_projected_guidance(
    pred_cond: torch.Tensor,  # U-Net conditional noise prediction [B, C, H, W]
    pred_uncond: torch.Tensor,  # U-Net unconditional noise prediction [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 15.0
):
    # Ensure same device
    pred_cond = pred_cond.to(pred_uncond.device)
    if pred_uncond is None:
        return pred_cond

    # Validate shapes
    if pred_cond.shape != pred_uncond.shape:
        raise ValueError(f"Shape mismatch: pred_cond {pred_cond.shape}, pred_uncond {pred_uncond.shape}")

    diff = pred_cond - pred_uncond

    if momentum_buffer is not None:
        diff = diff.to(pred_cond.device)
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average

    if norm_threshold > 0:
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(torch.ones_like(diff_norm), norm_threshold / diff_norm.clamp(min=1e-8))
        diff = diff * scale_factor

    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


def merge_latent_observations(
    current_latent: torch.Tensor,
    momentum_latent: torch.Tensor,
    rank: int = 2,
    momentum_gain: float = 0.30,
    agreement_tolerance: float = 0.50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Merge current and momentum observations in a joint SVD subspace.

    current_latent:
        Stable current latent, shape [B, C, H, W].

    momentum_latent:
        Momentum/history observation, shape [B, C, H, W].

    rank:
        Singular modes retained from each observation.

    momentum_gain:
        Strength of reliable momentum innovation.

    agreement_tolerance:
        Larger values accept more disagreement between observations.
    """

    if current_latent.shape != momentum_latent.shape:
        raise ValueError(
            f"Shape mismatch: {current_latent.shape} != {momentum_latent.shape}"
        )

    if current_latent.ndim != 4:
        raise ValueError("Expected tensors with shape [B, C, H, W].")

    original_dtype = current_latent.dtype
    batch_size, channels, height, width = current_latent.shape
    merged_latent = torch.empty_like(current_latent)

    for b in range(batch_size):
        current_matrix = current_latent[b].float().reshape(channels, -1)
        momentum_matrix = momentum_latent[b].float().reshape(channels, -1)

        # SVD of the two observations.
        U_current, _, Vh_current = torch.linalg.svd(
            current_matrix,
            full_matrices=False,
        )

        U_momentum, _, Vh_momentum = torch.linalg.svd(
            momentum_matrix,
            full_matrices=False,
        )

        actual_rank = min(
            rank,
            U_current.shape[1],
            U_momentum.shape[1],
            Vh_current.shape[0],
            Vh_momentum.shape[0],
        )

        # Concatenate left and right singular subspaces.
        U_joint = torch.cat(
            [
                U_current[:, :actual_rank],
                U_momentum[:, :actual_rank],
            ],
            dim=1,
        )

        V_joint = torch.cat(
            [
                Vh_current[:actual_rank].transpose(0, 1),
                Vh_momentum[:actual_rank].transpose(0, 1),
            ],
            dim=1,
        )

        # Polar-factor orthogonalization of the merged bases.
        P_u, _, Qh_u = torch.linalg.svd(U_joint, full_matrices=False)
        P_v, _, Qh_v = torch.linalg.svd(V_joint, full_matrices=False)

        U_merged = P_u @ Qh_u
        V_merged = P_v @ Qh_v

        # Change basis for both observations.
        current_coordinates = (
            U_merged.transpose(0, 1)
            @ current_matrix
            @ V_merged
        )

        momentum_coordinates = (
            U_merged.transpose(0, 1)
            @ momentum_matrix
            @ V_merged
        )

        # Momentum innovation in the common subspace.
        coordinate_difference = (
            momentum_coordinates - current_coordinates
        )

        # Accept momentum modes only when they are reasonably consistent
        # with the current observation.
        relative_difference = coordinate_difference.abs() / (
            current_coordinates.abs() + eps
        )

        agreement = torch.exp(
            -relative_difference / agreement_tolerance
        )

        merged_coordinates = (
            current_coordinates
            + momentum_gain
            * agreement
            * coordinate_difference
        )

        # Reconstruct the fused observation.
        merged_matrix = (
            U_merged
            @ merged_coordinates
            @ V_merged.transpose(0, 1)
        )

        merged_latent[b] = merged_matrix.reshape(
            channels,
            height,
            width,
        ).to(original_dtype)

    return merged_latent


class TrustRegionHistoryOrthogonalCFG:
    def __init__(self, window=6, rank=3, momentum=-0.6, eta=0.05, norm_ratio=0.35, eps=1e-8):
        self.window=window; self.rank=rank; self.momentum=momentum; self.eta=eta; self.norm_ratio=norm_ratio; self.eps=eps
        self.history=deque(maxlen=window); self.prev_m=None
        self.last_removed_ratio=0.0; self.last_trust_scale=1.0; self.last_scaled_update_ratio=0.0

    def reset(self):
        self.history.clear(); self.prev_m=None

    def update(self, noise_pred_uncond, noise_pred_text, guidance_scale):
        if noise_pred_uncond.shape!=noise_pred_text.shape: raise ValueError(f"Shape mismatch: {noise_pred_uncond.shape} != {noise_pred_text.shape}")
        dtype=noise_pred_text.dtype; u=noise_pred_uncond.float(); c=noise_pred_text.float(); B=u.shape[0]; w=float(guidance_scale)
        d=c-u
        if self.prev_m is None or self.prev_m.shape!=d.shape: self.prev_m=torch.zeros_like(d)
        m=d+self.momentum*self.prev_m
        self.prev_m=m.detach()
        flat_m=m.reshape(B,-1)
        if len(self.history)>=2:
            H=torch.stack([h.reshape(B,-1) for h in self.history],dim=1)
            _,_,Vh=torch.linalg.svd(H,full_matrices=False)
            r=min(self.rank,Vh.shape[1],Vh.shape[2])
            basis=Vh[:,:r,:]
            coeff=torch.einsum("bd,brd->br",flat_m,basis)
            persistent=torch.einsum("br,brd->bd",coeff,basis)
            safe_flat=flat_m-persistent+self.eta*persistent
            self.last_removed_ratio=(torch.linalg.vector_norm(persistent,dim=1)/(torch.linalg.vector_norm(flat_m,dim=1).clamp_min(self.eps))).mean().item()
        else:
            safe_flat=flat_m; self.last_removed_ratio=0.0

        c_flat=c.reshape(B,-1)
        scaled_update=w*safe_flat
        scaled_norm=torch.linalg.vector_norm(scaled_update,dim=1,keepdim=True).clamp_min(self.eps)
        ref_norm=torch.linalg.vector_norm(c_flat,dim=1,keepdim=True).clamp_min(self.eps)

        #trust_scale= torch.clamp(self.norm_ratio*ref_norm/scaled_norm,max=1.0)
        raw_ratio = scaled_norm / ref_norm  # q = ||w*s|| / ||c||

        target_ratio = torch.sqrt(self.norm_ratio * raw_ratio).clamp(max=self.norm_ratio)

        trust_scale = target_ratio / raw_ratio.clamp_min(self.eps)

        #print(trust_scale, ref_norm/scaled_norm )

        safe_flat=safe_flat*trust_scale

        #self.last_trust_scale=trust_scale.mean().item()
        #self.last_scaled_update_ratio=(torch.linalg.vector_norm(w*safe_flat,dim=1)/ref_norm.squeeze(1)).mean().item()

        safe_d=safe_flat.reshape_as(d)
        self.history.append(d.detach())
        out=u+w*safe_d
        return out.to(dtype)


if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # type: ignore

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionXLcovhisPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL with CovHiS guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.FloatTensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def pred_z0(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(sample.device)

        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_original_sample

    def pred_x0(self, latents, noise_pred, t, generator, device, prompt_embeds, output_type):
        pred_z0 = self.pred_z0(latents, noise_pred, t)
        pred_x0 = self.vae.decode(
            pred_z0 / self.vae.config.scaling_factor,
            return_dict=False,
            generator=generator
        )[0]
        #pred_x0, ____ = self.run_safety_checker(pred_x0, device, prompt_embeds.dtype)
        do_denormalize = [True] * pred_x0.shape[0]
        pred_x0 = self.image_processor.postprocess(pred_x0, output_type=output_type, do_denormalize=do_denormalize)

        return pred_x0

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def tpg_scale(self):
        return self._tpg_scale

    @property
    def do_token_perturbation_guidance(self):
        return self._tpg_scale > 0

    @property
    def tpg_applied_layers_index(self):
        return self._tpg_applied_layers_index

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 0.0,
        tpg_scale: float = 3.0,
        tpg_applied_layers_index: List[str] = ["d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23"],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        self._tpg_scale = tpg_scale
        self._tpg_applied_layers_index = tpg_applied_layers_index

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        #cfg

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        apg_eta = .7
        apg_norm_threshold  =  1 # higher value brings the saturation back
        apg_momentum  = -0.5
        apg_buffer = MomentumBuffer(apg_momentum)

        apg_o = MomentumBuffer(apg_momentum)

        bf1 = MomentumBuffer(0.1)

        bf2 = MomentumBuffer(0.1)


        tf1 = MomentumBuffer(0.1)

        tf2 = MomentumBuffer(0.1)

        tracker = AdamTracker(lr=0.003)

        tracker1 = AdamTracker(lr=0.009)

        bft = MomentumBuffer(0.81)

        detail_subspace = PersistentDetailSubspace(window=4,rank=3,ema_alpha=0.2,strength=-0.009, max_ratio=0.05, )
        tangent_refiner = TangentSubspaceTrustRegion( window=5, rank=3, strength=2.5, max_ratio=0.05, )

        covariance_refiner = TextCovarianceTangentGuidance( covariance_gain=4.5, eigen_power=1.5, strength=2.0, max_ratio=0.30, )

        svd_tone_transfer = SVDToneTransfer(tone_strength=0.9,mean_strength=0.8,max_singular_ratio=2.5,)
        tweedie_fusion = TweedieConsensusFusion( detail_strength=0.9, tone_strength=0.08, uncertainty_scale=2.0, max_ratio=0.25, )
        hocfg=TrustRegionHistoryOrthogonalCFG(window=6,rank=3,momentum=-0.6,eta=0.05,norm_ratio=0.35)

        ema_x = EMAK(alpha=0.8)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                #latents = detail_subspace.update(latents)
                # cfg
                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)

                # no
                else:
                    latent_model_input = latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds

                noise_pred = self.unet( latent_model_input,t, encoder_hidden_states=prompt_embeds,timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs, return_dict=False, )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                bft.update(noise_pred_uncond )
                nns = bft.running_average

                fdn = noise_pred_uncond - nns

                noise_pred_uncondp = tracker.update(noise_pred_uncond, nns)

                sc = optimized_scale(noise_pred_uncond, noise_pred_uncondp, noise_pred_uncond.shape[0])
                has_nan = torch.isnan(sc).any()
                has_inf = torch.isinf(sc).any()

                if has_nan or has_inf:
                    sc = torch.nan_to_num(sc, nan=1.0, posinf=1.0, neginf=1.0)

                noise_pred_uncondp = noise_pred_uncondp*sc

                noise_predo = hocfg.update(noise_pred_uncond,noise_pred_text,self.guidance_scale)
                #noise_predo = noise_pred_uncond  + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise_pred  = noise_predo - noise_pred_uncondp * 1e-2

                noise_pred = covariance_refiner.update( noise_pred_base=noise_pred, noise_pred_uncond=noise_pred_uncond, noise_pred_text=noise_pred_text, )

                #noise_pred_t=noise_pred_t
                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                latents = output

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
