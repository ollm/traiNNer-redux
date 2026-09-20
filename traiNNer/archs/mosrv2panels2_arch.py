from typing import Any

from traiNNer.archs.mosrv2multiscale_arch import MoSRv2MultiScale
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MoSRv2Panels2(MoSRv2MultiScale):
    """Panel-mask model that preserves the input R/B channels exactly."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["task"] = "panels2"
        super().__init__(**kwargs)


__all__ = ["MoSRv2Panels2"]
