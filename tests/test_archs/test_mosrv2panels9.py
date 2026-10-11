import pytest
import torch
from traiNNer.archs import build_network
from traiNNer.archs.mosrv2panels8_arch import MoSRv2Panels8
from traiNNer.archs.mosrv2panels9_arch import MoSRv2Panels9
from traiNNer.utils.panel_mask_util import reconstruct_panel_regions


def _small_model() -> MoSRv2Panels9:
    return MoSRv2Panels9(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        mask_head_blocks=1,
        detail_dim=8,
        detail_blocks=1,
    )


def _logits(mask: torch.Tensor) -> torch.Tensor:
    return torch.logit(mask.clamp(1e-4, 1 - 1e-4)).unsqueeze(0).unsqueeze(0)


def test_mosrv2panels9_preserves_shape_and_backpropagates() -> None:
    model = _small_model()
    input_tensor = torch.randn(1, 3, 33, 47, requires_grad=True)

    output = model(input_tensor)

    assert output.shape == input_tensor.shape
    assert torch.equal(output[:, 0], input_tensor[:, 0])
    assert torch.equal(output[:, 2], input_tensor[:, 2])
    output[:, 1].square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_mosrv2panels9_is_registered() -> None:
    model = build_network(
        {
            "type": "mosrv2panels9",
            "encoder_dims": (8, 12, 16),
            "encoder_blocks": (1, 1, 1),
            "decoder_blocks": (1, 1, 1),
            "num_downsamples": 2,
            "context_blocks": 1,
            "mask_head_blocks": 1,
            "detail_dim": 8,
            "detail_blocks": 1,
        }
    )

    assert isinstance(model, MoSRv2Panels9)


def test_mosrv2panels9_keeps_panels8_checkpoint_keys() -> None:
    panels8 = MoSRv2Panels8(
        encoder_dims=(8, 12, 16),
        encoder_blocks=(1, 1, 1),
        decoder_blocks=(1, 1, 1),
        num_downsamples=2,
        context_blocks=1,
        mask_head_blocks=1,
        detail_dim=8,
        detail_blocks=1,
    )

    assert _small_model().state_dict().keys() == panels8.state_dict().keys()


def test_reconstruction_fills_low_confidence_hole_inside_closed_panel() -> None:
    mask = torch.zeros(24, 32)
    mask[4:20, 5:27] = 1
    mask[4, 5:27] = 0
    mask[19, 5:27] = 0
    mask[4:20, 5] = 0
    mask[4:20, 26] = 0
    mask[11, 15] = 0.35

    reconstructed = reconstruct_panel_regions(_logits(mask))

    assert reconstructed.shape == (1, 1, 24, 32)
    assert reconstructed.dtype == torch.float32
    assert set(torch.unique(reconstructed).tolist()) <= {0.0, 1.0}
    assert reconstructed[0, 0, 11, 15] == 1
    assert reconstructed[0, 0, 4, 10] == 0
    assert reconstructed[0, 0, 0, 0] == 0


def test_reconstruction_preserves_two_panels_and_white_gutter() -> None:
    mask = torch.zeros(24, 40)
    mask[4:20, 3:17] = 1
    mask[4:20, 23:37] = 1

    reconstructed = reconstruct_panel_regions(_logits(mask))

    assert torch.equal(reconstructed[0, 0], mask)
    assert reconstructed[0, 0, 12, 19] == 0


def test_reconstruction_handles_incomplete_and_open_panels() -> None:
    incomplete_frame = torch.zeros(24, 32)
    incomplete_frame[4:20, 5:27] = 1
    incomplete_frame[4, 5:27] = 0
    incomplete_frame[19, 5:27] = 0
    incomplete_frame[4:20, 5] = 0
    incomplete_frame[4:20, 26] = 0
    incomplete_frame[4, 13:18] = 1

    open_panel = torch.zeros(24, 32)
    open_panel[3:21, 5:27] = 1
    open_panel[3, 5:27] = 0
    open_panel[20, 5:27] = 0
    open_panel[3:21, 5] = 0

    result = reconstruct_panel_regions(
        torch.cat((_logits(incomplete_frame), _logits(open_panel)))
    )

    assert result.shape == (2, 1, 24, 32)
    assert result[0, 0, 12, 10] == 1
    assert result[1, 0, 12, 10] == 1
    assert result[1, 0, 0, 0] == 0


def test_reconstruction_preserves_panel_touching_image_edge() -> None:
    mask = torch.zeros(24, 32)
    mask[0:18, 8:24] = 1
    mask[17, 8:24] = 0
    mask[0:18, 8] = 0
    mask[0:18, 23] = 0

    reconstructed = reconstruct_panel_regions(_logits(mask))

    assert reconstructed[0, 0, 0, 12] == 1
    assert reconstructed[0, 0, 0, 0] == 0


@pytest.mark.parametrize(
    "logits",
    [
        torch.empty(1, 3, 8, 8),
        torch.empty(1, 1, 0, 8),
    ],
)
def test_reconstruction_rejects_invalid_shapes(logits: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        reconstruct_panel_regions(logits)


def test_eval_postprocess_keeps_three_channel_interface() -> None:
    model = _small_model()
    output = torch.zeros(1, 3, 24, 32)
    output[:, 0] = 0.4
    output[:, 1] = -10
    output[:, 1, 5:19, 7:25] = 10
    output[:, 2] = 0.6
    output.requires_grad_()

    processed = model.post_process_inference(output)

    assert processed.shape == output.shape
    assert not processed.requires_grad
    assert torch.equal(processed[:, 0], output[:, 0])
    assert torch.equal(processed[:, 2], output[:, 2])
    assert set(torch.unique(processed[:, 1]).tolist()) <= {0.0, 1.0}
    assert processed[0, 1, 0, 0] == 0
    assert processed[0, 1, 12, 16] == 1
