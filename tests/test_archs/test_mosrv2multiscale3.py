import pytest
import torch

from traiNNer.archs.mosrv2multiscale3_arch import MoSRv2MultiScale3


def _small_model(**kwargs: object) -> MoSRv2MultiScale3:
    return MoSRv2MultiScale3(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        detail_dim=8,
        detail_blocks=1,
        **kwargs,
    )


def test_mosrv2multiscale3_refines_dynamic_restoration_and_backpropagates() -> None:
    model = _small_model(task="descreen")
    input_tensor = torch.randn(1, 3, 33, 47, requires_grad=True)

    output = model(input_tensor)

    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_mosrv2multiscale3_scales_with_deterministic_texture(scale: int) -> None:
    model = _small_model(scale=scale, task="noise").eval()
    input_tensor = torch.randn(1, 3, 17, 23)

    first = model(input_tensor)
    second = model(input_tensor)

    assert first.shape == (1, 3, 17 * scale, 23 * scale)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_mosrv2multiscale3_validates_task() -> None:
    with pytest.raises(ValueError, match="task"):
        _small_model(task="unknown")
