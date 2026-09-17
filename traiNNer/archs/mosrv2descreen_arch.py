from typing import Any

from traiNNer.archs.mosrv2multiscale_arch import MoSRv2MultiScale
from traiNNer.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class MoSRv2Descreen(MoSRv2MultiScale):
    """Compatibility wrapper for the RGB descreening task."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["task"] = "descreen"
        super().__init__(**kwargs)


__all__ = ["MoSRv2Descreen"]
