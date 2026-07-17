"""Root pytest conftest for the MLX test suites.

CI runs on virtualized macOS runners where Metal may be unavailable;
fall back to the MLX CPU backend so the unit suite still runs there.
"""

import mlx.core as mx

if not mx.metal.is_available():
    mx.set_default_device(mx.cpu)
