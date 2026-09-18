import torch

from traiNNer.archs.mosrv2noise_arch import MoSRv2Noise


def _small_model(**kwargs: object) -> MoSRv2Noise:
    return MoSRv2Noise(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        context_blocks=1,
        edge_refinement_blocks=1,
        **kwargs,
    )


def test_mosrv2noise_preserves_shape_and_gradients() -> None:
    model = _small_model()
    input_tensor = torch.randn(1, 3, 33, 47, requires_grad=True)

    output = model(input_tensor)

    assert output.shape == input_tensor.shape
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_mosrv2noise_is_stochastic_in_training() -> None:
    model = _small_model()
    input_tensor = torch.randn(1, 3, 32, 32)

    first = model(input_tensor)
    second = model(input_tensor)

    assert not torch.allclose(first, second)


def test_mosrv2noise_validates_strength() -> None:
    try:
        _small_model(noise_strength=0)
    except ValueError as error:
        assert "noise_strength" in str(error)
    else:
        raise AssertionError("noise_strength=0 should be rejected")


def test_mosrv2noise_validates_layer_count() -> None:
    try:
        _small_model(noise_layers=0)
    except ValueError as error:
        assert "noise_layers" in str(error)
    else:
        raise AssertionError("noise_layers=0 should be rejected")
