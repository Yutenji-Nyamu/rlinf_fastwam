"""Independent original EXPO-FT core; VLA and environment live in the driver."""

from .core import ExpoConfig, ExpoLearner, chunk_td_target, discounted_chunk_return

__all__ = ["ExpoConfig", "ExpoLearner", "chunk_td_target", "discounted_chunk_return"]
