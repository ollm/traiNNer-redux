import pytest
import torch

from traiNNer.archs.mosrv2descreen_arch import MoSRv2Descreen


def _small_model(**kwargs: object) -> MoSRv2Descreen:
    return MoSRv2Descreen(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        edge_refinement_blocks=1,
        **kwargs,
    )


@pytest.mark.parametrize("height,width", [(32, 32), (33, 47), (48, 65)])
def test_mosrv2descreen_preserves_dynamic_shape(height: int, width: int) -> None:
    model = _small_model()
    input_tensor = torch.randn(1, 3, height, width, requires_grad=True)

    output = model(input_tensor)

    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_mosrv2descreen_validates_scale_lists() -> None:
    with pytest.raises(ValueError, match="encoder_dims"):
        _small_model(encoder_dims=(8, 12))

    with pytest.raises(ValueError, match="decoder_blocks"):
        _small_model(decoder_blocks=(1, 1))

    with pytest.raises(ValueError, match="scale=1"):
        _small_model(scale=2)
