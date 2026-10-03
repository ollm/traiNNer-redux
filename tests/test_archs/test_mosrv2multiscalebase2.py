from typing import TypedDict, Unpack

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from traiNNer.archs.mosrv2multiscale_arch import MoSRv2MultiScale
from traiNNer.archs.mosrv2multiscalebase2_arch import MoSRv2MultiScaleBase2


class _ModelOptions(TypedDict, total=False):
    frequency_guidance: bool
    frequency_kernel_size: int
    scale: int
    task: str
    zero_init_residual: bool


def _small_model(**kwargs: Unpack[_ModelOptions]) -> MoSRv2MultiScaleBase2:
    return MoSRv2MultiScaleBase2(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        **kwargs,
    )


def test_multiscalebase2_default_matches_multiscale_descreen() -> None:
    torch.manual_seed(1)
    baseline = MoSRv2MultiScale(
        task="descreen",
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
    ).eval()
    torch.manual_seed(1)
    model = MoSRv2MultiScaleBase2(
        task="descreen",
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
    ).eval()
    input_tensor = torch.randn(1, 3, 33, 47)

    assert torch.equal(model(input_tensor), baseline(input_tensor))


def test_multiscalebase2_zero_initialized_residual_starts_as_identity() -> None:
    model = _small_model(zero_init_residual=True).eval()
    input_tensor = torch.randn(1, 3, 33, 47)

    assert torch.equal(model(input_tensor), input_tensor)


def test_multiscalebase2_zero_initialized_residual_preserves_interpolation_base() -> (
    None
):
    model = _small_model(scale=2, zero_init_residual=True).eval()
    input_tensor = torch.randn(1, 3, 32, 48)
    expected = F.interpolate(
        input_tensor, scale_factor=2, mode="bilinear", align_corners=False
    )

    assert torch.equal(model(input_tensor), expected)


def test_multiscalebase2_frequency_guidance_exposes_features_and_gradients() -> None:
    model = _small_model(frequency_guidance=True)
    input_tensor = torch.randn(1, 3, 33, 47, requires_grad=True)

    output, features = model.forward_features(input_tensor)

    assert model.feature_dim == 8
    assert output.shape == input_tensor.shape
    assert features.shape == (1, 8, 33, 47)
    (output.square().mean() + features.square().mean()).backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_multiscalebase2_rejects_unsupported_options() -> None:
    with pytest.raises(ValueError, match="task"):
        _small_model(task="panels")
    with pytest.raises(ValueError, match="odd integer"):
        _small_model(frequency_kernel_size=2)
    with pytest.raises(ValueError, match="odd integer"):
        _small_model(frequency_kernel_size=1)
