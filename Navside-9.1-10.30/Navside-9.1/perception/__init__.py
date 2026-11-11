from __future__ import annotations


def create_depth_perception(backend: str = "realsense", realsense_config_path: str = "", **kwargs):
    """Create a depth camera backend without importing hardware SDKs eagerly."""
    normalized = backend.strip().lower().replace("-", "_")
    if normalized in ("zed", "zed_mini", "zedmini"):
        from .zed_depth_perception import ZedDepthPerception

        return ZedDepthPerception(**kwargs)
    if normalized in ("realsense", "d455", "d455f"):
        from .realsense_depth_perception import RealSenseDepthPerception

        return RealSenseDepthPerception(realsense_config_path=realsense_config_path, **kwargs)
    if normalized in ("orbbec", "gemini", "gemini336l", "gemini_336l", "336l"):
        from .orbbec_depth_perception import OrbbecDepthPerception

        return OrbbecDepthPerception(**kwargs)
    raise ValueError(
        f"unsupported depth backend: {backend!r}; "
        "expected 'zed', 'realsense', or 'orbbec'"
    )
