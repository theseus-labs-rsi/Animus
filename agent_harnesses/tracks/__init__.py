"""两条赛道的执行适配器。"""

from .base import TrackAdapter
from .memory import MemoryTrackAdapter
from .native import NativeTrackAdapter


def get_track_adapter(track: str) -> TrackAdapter:
    if track == "native":
        return NativeTrackAdapter()
    if track == "memory":
        return MemoryTrackAdapter()
    raise ValueError(f"unknown track: {track}")


__all__ = [
    "TrackAdapter",
    "NativeTrackAdapter",
    "MemoryTrackAdapter",
    "get_track_adapter",
]

