from diffusers import StableDiffusion3Pipeline


import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3IPAdapterMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    logger, 
    EXAMPLE_DOC_STRING,
    calculate_shift, 
    retrieve_timesteps,
)

import torch


import torch.nn as nn


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
 
import copy

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
 



def optimized_scale(noise_pred_text,noise_pred_uncond, batch_size):
    
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

class StableDiffusion3covhisPipeline(StableDiffusion3Pipeline):

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,

        ## Tangential Scailing Guidance specific parameters
        t_guidance_scale: float = 1.0,  # Scale for TGS
        r_guidance_scale: float = 1.0,  # Scale for radial guidance

        ## Apply range for each scaling
        sta_tpd: int = 1000,  # Start step for tangential scaling
        end_tpd: int = 0,  # End step for tangential scaling
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
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
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
                emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
                `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
                a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
            skip_guidance_layers (`List[int]`, *optional*):
                A list of integers that specify layers to skip during guidance. If not provided, all layers will be
                used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
                Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
            skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
                `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
                with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
                with a scale of `1`.
            skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
                `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
            skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
                `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
            mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
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

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)
        sc1 = self._guidance_scale
        sc2 = 16
        # 7. Denoising loop
        ema_x = EMAK(alpha=0.8)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noy, nop = project( noise_pred_uncond, noise_pred_text)
                    #### eta pore kaje lagbe ######
                    noise_pred_aux = noise_pred_uncond + 5.5 * (noise_pred_text - noise_pred_uncond) - .02 * (noy)
                    ############################# 
                    moy, mop = project( noise_pred_text, noise_pred_uncond)
                    w1 = 0.19 #eta kom thakte hbe
                    w2 = 1- w1
                    noise_pred_uncond = w1 * noise_pred_uncond + w2 * (moy-mop)  
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) - 0.012 * (noy) 
                    #### uporer... 2line convert....uddessho details rakha
                    ### manifold high saturation hole class uddhar koro.
                    if self.guidance_scale>7:
                        noy, nop = project( noise_pred_uncond, noise_pred_text )
                        noise_predf = noy + self.guidance_scale *(noise_pred_text  - noy)
                        w1 = 0.6
                        w2 = 1-w1
                        noise_pred = w1* noise_pred + noise_predf*w2
                    #### manifold e high saturation hole class uddhar koro. sheta je kono cfg scale. 
                    ###important for any cfg
                    df = noise_pred  - ema_x(noise_pred)
                    w1 = 0.99
                    w2 = 1-w1
                    noise_pred   = noise_pred *w1 + df*w2
                    ### important for high cfg
                    if self.guidance_scale>7:
                        noise_pred = simple_distribution_transfer(noise_pred, noise_pred_aux)
                    
                    # df = noise_pred  - ema_x(noise_pred)
                    # w1 = 0.95
                    # w2 = 1-w1
                    # noise_pred   = noise_pred *w1 + df*w2
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                output = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                latents = output

                # [NOTE] Apple MPS Bug -- Don't need to care 
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
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
