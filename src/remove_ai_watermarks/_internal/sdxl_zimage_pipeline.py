"""The sdxl-zimage recipe with an SDXL global stage.

Only the global regeneration model changes. The face stage is inherited verbatim
from :class:`TwoStageZImagePipeline` -- same YuNet detection, same SAM masks, same
Z-Image Turbo repair of the original crops, same feathered compositing -- so a
change there cannot silently diverge between the profiles.

Three pieces cannot be shared, because they are bound to the architecture: the
ControlNet, the four-step distillation LoRA, and the sampler. Strength is bound to
it too, which is the part that is easy to miss: an SDXL global pass leaves SynthID
at the strength Qwen needs. See ``watermark_profiles.SDXL_ZIMAGE_OPENAI_STRENGTH``.
"""

# Diffusers and torch expose mostly untyped tensor APIs. Keep the relaxation local
# to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportPrivateImportUsage=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from remove_ai_watermarks._internal.two_stage_pipeline import (
    _GLOBAL_NEGATIVE,
    _GLOBAL_PROMPT,
    TwoStageZImagePipeline,
    _target_size,
    build_canny_control_image,
    diffusers_vae_roundtrip,
    requested_steps,
)
from remove_ai_watermarks._internal.watermark_profiles import (
    CONTROLNET_CANNY_MODEL,
    SDXL_LIGHTNING_MODEL_ID,
    SDXL_LIGHTNING_PATTERN,
    SDXL_MODEL_ID,
)

log = logging.getLogger(__name__)

SDXL_VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
# SDXL aligns to an 8-pixel latent grid, against Qwen's 16.
_LATENT_GRID = 8
# SDXL-Lightning's own four-step distillation, at the strength its authors document.
# The reference graph loads the Qwen LoRA at 0.8; carrying that number to a different
# LoRA on a different architecture would be imitation, not parity.
SDXL_STEPS = 4
# Below this floor the global stack streams its weights from CPU layer by layer; at or
# above it the stack stays resident. The fp16 UNet, Canny ControlNet, both text
# encoders and the VAE are ~8.8 GiB of weights, and the UNet and the ControlNet run
# together on every step, so model-level offload cannot get under an 8 GiB card.
# Benchmark in docs/module-internals.md, "CPU offload".
RESIDENT_SDXL_GLOBAL_MIN_VRAM_GIB = 12.0


def sdxl_target_size(width: int, height: int) -> tuple[int, int]:
    """Floor dimensions to SDXL's latent grid without changing aspect."""
    return _target_size(width, height, _LATENT_GRID)


def resolve_sdxl_global_residency(
    requested: bool | None,
    *,
    total_memory_gib: float,
) -> bool:
    """Keep the SDXL global stack resident when explicitly requested or safely sized."""
    if requested is not None:
        return requested
    return total_memory_gib >= RESIDENT_SDXL_GLOBAL_MIN_VRAM_GIB


@dataclass
class SdxlZImagePipeline(TwoStageZImagePipeline):
    """Lazy runtime for the SDXL global stage plus the inherited face stage."""

    profile_name: ClassVar[str] = "sdxl-zimage"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sdxl_pipe: Any = None

    def _keep_global_models_resident(self) -> bool:
        return resolve_sdxl_global_residency(
            self.keep_global_models_on_device,
            total_memory_gib=self._total_vram_gib(),
        )

    def _load_global(self) -> Any:
        if self._sdxl_pipe is not None:
            return self._sdxl_pipe
        self._require_cuda()
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            EulerDiscreteScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
        from huggingface_hub import hf_hub_download

        self._progress("Loading SDXL, Lightning LoRA, and Canny ControlNet...")
        token = {"token": self.hf_token} if self.hf_token else {}
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_MODEL, torch_dtype=torch.float16, **token)
        vae = AutoencoderKL.from_pretrained(SDXL_VAE_MODEL_ID, torch_dtype=torch.float16, **token)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            SDXL_MODEL_ID,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            add_watermarker=False,
            **token,
        )
        resident = self._keep_global_models_resident()
        if resident:
            pipe = pipe.to(self.device)
        pipe.load_lora_weights(hf_hub_download(SDXL_LIGHTNING_MODEL_ID, SDXL_LIGHTNING_PATTERN, **token))
        pipe.fuse_lora()
        # SDXL-Lightning is distilled against trailing timestep spacing.
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        if not resident:
            # After the fuse: the offload hooks keep their own CPU copy of each weight
            # and stream that copy, so fusing later would never reach what runs.
            self._progress("Streaming the SDXL stack from CPU (card below the residency floor)...")
            pipe.enable_sequential_cpu_offload(device=self.device)
        self._sdxl_pipe = pipe
        return pipe

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        import torch

        pipe = self._load_global()
        target = sdxl_target_size(image.width, image.height)
        prepared = image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
        control = build_canny_control_image(prepared)
        steps = requested_steps(SDXL_STEPS, strength)
        self._progress(f"Running SDXL Canny pass: strength={strength:.4f}, steps={SDXL_STEPS} of {steps}...")
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        result = pipe(
            prompt=_GLOBAL_PROMPT,
            negative_prompt=_GLOBAL_NEGATIVE,
            image=prepared,
            control_image=control,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

    def _vae_roundtrip(self, image: Image.Image) -> Image.Image:
        """Reconstruct source pixels through the already loaded SDXL VAE."""
        pipe = self._load_global()
        return diffusers_vae_roundtrip(
            pipe,
            image,
            grid=_LATENT_GRID,
            device=self.device,
            dtype=self.torch_dtype,
            profile_name=self.profile_name,
        )
