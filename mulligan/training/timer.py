#!/usr/bin/env python3
"""
Timing utilities for profiling training loops and other operations.
"""

import time

import torch


class TrainingTimer:
    """Simple timing utility using context manager pattern for profiling training loops."""

    def __init__(self, cuda_sync: bool = False):
        self.timings = {}
        self.counts = {}
        self._current_key = None
        self._start_time = None
        self._cuda_sync = cuda_sync

    def __call__(self, key: str):
        """Use as context manager: with timer('forward'): ..."""
        self._current_key = key
        return self

    def __enter__(self):
        """Start timing when entering context."""
        if self._cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        """Stop timing when exiting context."""
        if self._cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_time
        if self._current_key not in self.timings:
            self.timings[self._current_key] = 0.0
            self.counts[self._current_key] = 0
        self.timings[self._current_key] += elapsed
        self.counts[self._current_key] += 1
        self._current_key = None
        self._start_time = None

    def get_stats(self) -> dict[str, float]:
        """Get average time per operation in seconds."""
        return {
            key: self.timings[key] / self.counts[key]
            for key in self.timings
            if self.counts[key] > 0
        }

    def reset(self):
        """Reset all timing statistics."""
        self.timings.clear()
        self.counts.clear()

    def print_stats(self, prefix: str = ""):
        """Print formatted timing statistics on a single line."""
        stats = self.get_stats()
        if not stats:
            return

        total_time = sum(stats.values())
        parts = [f"{key}: {avg_time * 1000:.1f}ms" for key, avg_time in stats.items()]
        parts.append(f"total: {total_time * 1000:.1f}ms")
        print(f"{prefix}Timing: {' | '.join(parts)}")
