import timeit

import torch
import torch.cuda.nvtx as nvtx


class ProfilingRange:
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            nvtx.range_push(self.name)
            torch.cuda.synchronize()
        self._start = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            nvtx.range_pop()
        self.elapsed = timeit.default_timer() - self._start
        return False


def profiling_range(name: str) -> ProfilingRange:
    return ProfilingRange(name)
