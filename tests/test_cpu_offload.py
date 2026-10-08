"""Unit tests for how far --cpu-offload travels, per profile.

The flag sets two residency fields. The face one is read by the shared base and
reaches every profile; the global one is read by ``QwenZImagePipeline`` and
``SdxlZImagePipeline``, so on ``chroma-zimage`` it never touches the stack the user
is trying to fit. ``WatermarkRemover`` says so before the model load.

Both halves are exercised with an uninitialized remover, so the core CI matrix
needs no diffusion dependency, model download, or GPU. The SDXL loader placement is
driven through stubbed Diffusers constructors and skips without the diffusion extra.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from remove_ai_watermarks._internal.sdxl_zimage_pipeline import (
    RESIDENT_SDXL_GLOBAL_MIN_VRAM_GIB,
    resolve_sdxl_global_residency,
)
from remove_ai_watermarks._internal.watermark_profiles import (
    AUTO_PROFILE,
    CHROMA_ZIMAGE_PROFILE,
    GLOBAL_OFFLOAD_PROFILES,
    PROFILE_CHOICES,
    QWEN_ZIMAGE_PROFILE,
    SDXL_ZIMAGE_PROFILE,
    global_offload_supported,
)
from remove_ai_watermarks._internal.watermark_remover import WatermarkRemover

OFFLOAD_SUPPORT_CASES = (
    (QWEN_ZIMAGE_PROFILE, True),
    (CHROMA_ZIMAGE_PROFILE, False),
    (SDXL_ZIMAGE_PROFILE, True),
    # Auto preloads its concrete Qwen fallback before an image vendor is known.
    (AUTO_PROFILE, True),
)


def _remover(profile: str, cpu_offload: bool) -> WatermarkRemover:
    remover = WatermarkRemover.__new__(WatermarkRemover)
    remover.configured_profile = profile
    remover.model_profile = QWEN_ZIMAGE_PROFILE if profile == AUTO_PROFILE else profile
    remover.cpu_offload = cpu_offload
    remover.device = "cuda"
    remover.torch_dtype = None
    remover.hf_token = None
    remover.controlnet_conditioning_scale = 1.0
    remover._progress_callback = None
    remover._qwen_zimage_pipeline = None
    return remover


class TestGlobalOffloadSupport:
    @pytest.mark.parametrize(
        ("profile", "expected"),
        OFFLOAD_SUPPORT_CASES,
    )
    def test_loaded_profile_or_preload_fallback_reports_support(self, profile: str, expected: bool) -> None:
        assert global_offload_supported(profile) is expected

    def test_cases_cover_every_profile(self) -> None:
        assert {profile for profile, _expected in OFFLOAD_SUPPORT_CASES} == set(PROFILE_CHOICES)

    def test_the_underscore_spelling_resolves(self) -> None:
        assert global_offload_supported("qwen_zimage") is True

    def test_the_set_names_a_profile_that_exists(self) -> None:
        # A renamed profile must not leave the flag silently unsupported everywhere.
        assert set(PROFILE_CHOICES) >= GLOBAL_OFFLOAD_PROFILES


class TestUnsupportedProfileWarns:
    def test_it_names_the_profile_and_the_flag(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _remover(CHROMA_ZIMAGE_PROFILE, cpu_offload=True)._warn_if_global_offload_unsupported()
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert "--cpu-offload" in message
        assert CHROMA_ZIMAGE_PROFILE in message

    @pytest.mark.parametrize("profile", [QWEN_ZIMAGE_PROFILE, SDXL_ZIMAGE_PROFILE])
    def test_a_profile_that_streams_its_global_stack_is_quiet(
        self, profile: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _remover(profile, cpu_offload=True)._warn_if_global_offload_unsupported()
        assert caplog.records == []

    def test_a_run_that_did_not_ask_for_offload_is_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _remover(CHROMA_ZIMAGE_PROFILE, cpu_offload=False)._warn_if_global_offload_unsupported()
        assert caplog.records == []

    def test_preloaded_auto_uses_qwen_offload_without_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from remove_ai_watermarks._internal import qwen_zimage_pipeline

        captured: dict[str, object] = {}

        class Recorder:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def preload(self, *, global_only: bool = False) -> None:
                captured["global_only"] = global_only

        monkeypatch.setattr(qwen_zimage_pipeline, "QwenZImagePipeline", Recorder)
        with caplog.at_level(logging.WARNING):
            _remover(AUTO_PROFILE, cpu_offload=True).preload(global_only=True)

        assert caplog.records == []
        assert captured["keep_global_models_on_device"] is False
        assert captured["global_only"] is True

    def test_the_load_path_asks_before_building_the_stack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The warning has to reach the user ahead of the download, and it has to see
        # the profile auto resolved to, so the seam is the load and not __init__.
        remover = _remover(CHROMA_ZIMAGE_PROFILE, cpu_offload=True)
        order: list[str] = []
        monkeypatch.setattr(
            WatermarkRemover,
            "_warn_if_global_offload_unsupported",
            lambda self: order.append("warned"),
        )

        def _explode(*_args: object, **_kwargs: object) -> object:
            order.append("loaded")
            raise ImportError("no diffusion dependency in this environment")

        monkeypatch.setattr(
            "remove_ai_watermarks._internal.chroma_zimage_pipeline.ChromaZImagePipeline",
            _explode,
        )
        with pytest.raises(ImportError):
            remover._load_qwen_zimage_pipeline()
        assert order == ["warned", "loaded"]


class TestSdxlGlobalResidency:
    @pytest.mark.parametrize(
        ("requested", "total_memory_gib", "expected"),
        [
            # Unset: the card decides, and an 8 GiB card cannot hold the stack.
            (None, 7.6, False),
            (None, RESIDENT_SDXL_GLOBAL_MIN_VRAM_GIB, True),
            (None, 23.5, True),
            # An unreadable card reports 0.0 and must stream rather than spill.
            (None, 0.0, False),
            # Explicit requests win in both directions.
            (False, 79.2, False),
            (True, 7.6, True),
        ],
    )
    def test_request_wins_and_the_card_decides_otherwise(
        self, requested: bool | None, total_memory_gib: float, expected: bool
    ) -> None:
        assert resolve_sdxl_global_residency(requested, total_memory_gib=total_memory_gib) is expected


class _RecordingSdxlPipe:
    """Stands in for the Diffusers SDXL pipeline and records placement calls in order."""

    def __init__(self, calls: list[Any]) -> None:
        self.calls = calls
        self.scheduler = MagicMock()

    def to(self, device: str) -> _RecordingSdxlPipe:
        self.calls.append(("to", device))
        return self

    def load_lora_weights(self, _path: str) -> None:
        self.calls.append(("load_lora_weights",))

    def fuse_lora(self) -> None:
        self.calls.append(("fuse_lora",))

    def enable_sequential_cpu_offload(self, *, device: str) -> None:
        self.calls.append(("enable_sequential_cpu_offload", device))


class TestSdxlGlobalPlacement:
    @staticmethod
    def _load(monkeypatch: pytest.MonkeyPatch, *, requested: bool | None, total_memory_gib: float) -> list[Any]:
        diffusers = pytest.importorskip("diffusers")
        pytest.importorskip("torch")
        from remove_ai_watermarks._internal.sdxl_zimage_pipeline import SdxlZImagePipeline

        calls: list[Any] = []
        monkeypatch.setattr(diffusers.ControlNetModel, "from_pretrained", lambda *a, **k: MagicMock())
        monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", lambda *a, **k: MagicMock())
        monkeypatch.setattr(
            diffusers.StableDiffusionXLControlNetImg2ImgPipeline,
            "from_pretrained",
            lambda *a, **k: _RecordingSdxlPipe(calls),
        )
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "lora.safetensors")
        monkeypatch.setattr(diffusers.EulerDiscreteScheduler, "from_config", lambda *a, **k: MagicMock())

        pipeline = SdxlZImagePipeline(device="cuda", torch_dtype=None, keep_global_models_on_device=requested)
        monkeypatch.setattr(type(pipeline), "_require_cuda", lambda self: None)
        monkeypatch.setattr(type(pipeline), "_total_vram_gib", lambda self: total_memory_gib)
        pipeline._load_global()
        return calls

    def test_a_small_card_streams_the_fused_stack_instead_of_moving_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._load(monkeypatch, requested=None, total_memory_gib=7.6)
        # The offload hooks stream their own copy of each weight, so they must follow the fuse.
        assert calls == [("load_lora_weights",), ("fuse_lora",), ("enable_sequential_cpu_offload", "cuda")]

    def test_a_large_card_keeps_the_stack_resident_as_before(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._load(monkeypatch, requested=None, total_memory_gib=23.5)
        assert calls == [("to", "cuda"), ("load_lora_weights",), ("fuse_lora",)]

    def test_cpu_offload_streams_even_on_a_large_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._load(monkeypatch, requested=False, total_memory_gib=79.2)
        assert ("to", "cuda") not in calls
        assert calls[-1] == ("enable_sequential_cpu_offload", "cuda")

    def test_cpu_offload_reaches_the_sdxl_stack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from remove_ai_watermarks._internal import sdxl_zimage_pipeline

        captured: dict[str, object] = {}

        class Recorder:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(sdxl_zimage_pipeline, "SdxlZImagePipeline", Recorder)
        _remover(SDXL_ZIMAGE_PROFILE, cpu_offload=True)._load_qwen_zimage_pipeline()
        assert captured["keep_global_models_on_device"] is False
