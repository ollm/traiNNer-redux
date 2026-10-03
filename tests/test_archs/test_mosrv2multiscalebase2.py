from typing import TypedDict, Unpack

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from traiNNer.archs.mosrv2multiscale_arch import MoSRv2MultiScale
from traiNNer.archs.mosrv2multiscalebase2_arch import MoSRv2MultiScaleBase2


class _ModelOptions(TypedDict, total=False):
    anti_alias_downsample: bool
    frequency_guidance: bool
    frequency_kernel_size: int
    frequency_kernel_sizes: tuple[int, ...]
    global_context_dilations: tuple[int, ...]
    noise_input_conditioning: bool
    scale: int
    task: str
    zero_init_noise: bool
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


def test_multiscalebase2_supports_multiband_anti_alias_global_context() -> None:
    model = _small_model(
        frequency_guidance=True,
        frequency_kernel_sizes=(3, 7, 15),
        anti_alias_downsample=True,
        global_context_dilations=(1, 2, 4),
    )
    input_tensor = torch.randn(1, 3, 33, 47, requires_grad=True)

    output = model(input_tensor)

    assert model.stem.in_channels == 12
    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_multiscalebase2_noise_uses_input_conditioning_by_default() -> None:
    model = _small_model(task="noise", zero_init_noise=True)
    parameters = dict(model.named_parameters())

    assert "noise_injection.source_condition.0.weight" in parameters
    assert (
        torch.count_nonzero(
            parameters["noise_injection.injection.noise_features.4.weight"]
        )
        == 0
    )


def test_multiscalebase2_can_disable_input_conditioned_noise() -> None:
    model = _small_model(task="noise", noise_input_conditioning=False)

    assert not any(
        name.startswith("noise_injection.source_condition")
        for name in model.state_dict()
    )


def test_multiscalebase2_rejects_unsupported_options() -> None:
    with pytest.raises(ValueError, match="task"):
        _small_model(task="panels")
    with pytest.raises(ValueError, match="odd integer"):
        _small_model(frequency_kernel_size=2)
    with pytest.raises(ValueError, match="odd integer"):
        _small_model(frequency_kernel_size=1)
    with pytest.raises(ValueError, match="frequency_kernel_sizes"):
        _small_model(frequency_kernel_sizes=(3, 4))
    with pytest.raises(ValueError, match="global_context_dilations"):
        _small_model(global_context_dilations=(0,))
