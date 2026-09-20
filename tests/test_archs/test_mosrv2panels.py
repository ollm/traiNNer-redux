import pytest
import torch

from traiNNer.archs.mosrv2multiscale_arch import MoSRv2MultiScale
from traiNNer.archs.mosrv2panels_arch import MoSRv2Panels
from traiNNer.archs.mosrv2panels2_arch import MoSRv2Panels2


def _small_model(**kwargs: object) -> MoSRv2Panels:
    return MoSRv2Panels(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        context_blocks=1,
        edge_refinement_blocks=1,
        **kwargs,
    )


@pytest.mark.parametrize("height,width", [(32, 32), (33, 47), (48, 65)])
def test_mosrv2panels_preserves_dynamic_shape(height: int, width: int) -> None:
    model = _small_model()
    input_tensor = torch.randn(1, 3, height, width, requires_grad=True)

    output = model(input_tensor)

    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_mosrv2panels_validates_scale_lists() -> None:
    with pytest.raises(ValueError, match="encoder_dims"):
        _small_model(encoder_dims=(8, 12))

    with pytest.raises(ValueError, match="decoder_blocks"):
        _small_model(decoder_blocks=(1, 1))

    with pytest.raises(ValueError, match="scale=1"):
        _small_model(scale=2)


def test_mosrv2multiscale_validates_task() -> None:
    with pytest.raises(ValueError, match="task"):
        MoSRv2MultiScale(task="unknown")


def test_mosrv2panels2_preserves_input_rgb_channels() -> None:
    model = MoSRv2Panels2(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        context_blocks=1,
        edge_refinement_blocks=1,
    )
    input_tensor = torch.randn(1, 3, 32, 32, requires_grad=True)

    output = model(input_tensor)

    assert torch.equal(output[:, 0], input_tensor[:, 0])
    assert torch.equal(output[:, 2], input_tensor[:, 2])
    assert output.shape == input_tensor.shape
    output[:, 1].square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_mosrv2panels2_requires_scale_one() -> None:
    with pytest.raises(ValueError, match="scale=1"):
        MoSRv2Panels2(scale=2)


def test_mosrv2panels2_configures_mask_head_depth() -> None:
    model = MoSRv2Panels2(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        mask_head_blocks=3,
    )

    assert len(model.to_mask) == 7
