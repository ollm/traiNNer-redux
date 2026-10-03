from typing import TypedDict, Unpack

import pytest
import torch
from torch import nn
from traiNNer.archs.mosrv2multiscale3_arch import MoSRv2MultiScale3
from traiNNer.archs.mosrv2multiscale4_arch import MoSRv2MultiScale4


class _ModelOptions(TypedDict, total=False):
    pattern_alignment_dim: int
    pattern_alignment_enabled: bool | None
    pattern_max_source_angle_degrees: float
    pattern_target_angle_degrees: float
    task: str


def _small_model(**kwargs: Unpack[_ModelOptions]) -> MoSRv2MultiScale4:
    return MoSRv2MultiScale4(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        detail_dim=8,
        detail_blocks=1,
        **kwargs,
    )


def test_mosrv2multiscale4_disabled_matches_multiscale3() -> None:
    torch.manual_seed(1)
    baseline = MoSRv2MultiScale3(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        detail_dim=8,
        detail_blocks=1,
    ).eval()
    torch.manual_seed(1)
    model = MoSRv2MultiScale4(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        detail_dim=8,
        detail_blocks=1,
    ).eval()
    input_tensor = torch.randn(1, 3, 33, 47)

    assert model.pattern_aligner is None
    assert torch.equal(model(input_tensor), baseline(input_tensor))


def test_mosrv2multiscale4_aligns_patterns_and_backpropagates() -> None:
    model = _small_model(
        task="noise,pattern",
        pattern_target_angle_degrees=30,
    )
    input_tensor = torch.randn(1, 3, 33, 47, requires_grad=True)

    output = model(input_tensor)

    assert model.pattern_aligner is not None
    assert model.tasks == frozenset(("noise", "pattern"))
    assert not isinstance(model.noise_injection, nn.Identity)
    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()
    assert model.pattern_aligner.source_angle_head.weight.grad is not None


def test_mosrv2multiscale4_validates_pattern_alignment_options() -> None:
    with pytest.raises(ValueError, match="positive"):
        _small_model(pattern_alignment_dim=0)
    with pytest.raises(ValueError, match=r"\[-180, 180\]"):
        _small_model(pattern_target_angle_degrees=181)
    with pytest.raises(ValueError, match=r"\(0, 180\]"):
        _small_model(pattern_max_source_angle_degrees=0)


@pytest.mark.parametrize("task", ("unknown", "noise,unknown", "noise,noise", ","))
def test_mosrv2multiscale4_validates_combined_tasks(task: str) -> None:
    with pytest.raises(ValueError, match="comma-separated combination"):
        _small_model(task=task)
