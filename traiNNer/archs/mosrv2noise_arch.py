from typing import Any

from traiNNer.archs.mosrv2multiscale_arch import MoSRv2MultiScale
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MoSRv2Noise(MoSRv2MultiScale):
    """MoSRv2 variant with a deterministic, conditioned noise residual."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["task"] = "noise"
        super().__init__(**kwargs)


__all__ = ["MoSRv2Noise"]
