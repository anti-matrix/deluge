"""Numba's ``njit`` for the integer and boolean kernels (Round 11), or, without numba, a
stand-in that leaves the function as plain Python (same results, much slower)."""

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is a dependency; this keeps the sim importable without it
    def njit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)

__all__ = ["njit"]
