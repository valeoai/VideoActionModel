# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name,function-redefined,unused-argument
# flake8: noqa
"""Wrapper for NVTX (NVIDIA Tools Extension) with dummy fallbacks."""
import contextlib
from itertools import cycle
from typing import Any

from lightning.pytorch.profilers import PassThroughProfiler, Profiler

try:
    from nvtx import annotate, enabled, end_range, get_domain, mark, pop_range, push_range, start_range
except ImportError:
    nvtx = None

    def enabled():
        return False


__all__ = [
    "annotate",
    "get_domain",
    "get_domain_color",
    "mark",
    "push_range",
    "pop_range",
    "start_range",
    "end_range",
    "enabled",
]

COLORS = (
    "green",
    "blue",
    "yellow",
    "purple",
    "rapids",
    "cyan",
    "red",
    "white",
    "darkgreen",
    "orange",
)
COLOR_CYCLE = cycle(COLORS)
domain_to_color: dict[str, str] = {}


def get_domain_color(domain):
    if domain in domain_to_color:
        return domain_to_color[domain]
    color = next(COLOR_CYCLE)
    domain_to_color[domain] = color
    return color


if enabled():

    class NVTXProfiler(Profiler):
        """"Pytorch Ligthning adapter for NVTX annotations."""

        def __init__(self, color=None, domain=None, category=None, dirpath=None, filename=None):
            super().__init__(dirpath, filename)
            self.color = color
            self.domain = domain
            self.category = category
            self.ranges: dict[str, Any] = {}

        def profile(self, action_name):
            return annotate(action_name, color=self.color, domain=self.domain, category=self.category)

        def start(self, action_name):
            if action_name not in self.ranges:
                self.ranges[action_name] = start_range(
                    action_name, color=self.color, domain=self.domain, category=self.category
                )
            return self.ranges[action_name]

        def stop(self, action_name):
            action_range = self.ranges.pop(action_name, None)
            if action_range is not None:
                end_range(action_range)

else:

    NVTXProfiler: Profiler = PassThroughProfiler  # type: ignore[no-redef,assignment]

    class annotate(contextlib.nullcontext):  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            super().__init__()

        def __call__(self, func):
            return func

    def mark(message=None, color=None, domain=None, category=None, payload=None):
        pass

    def push_range(message=None, color=None, domain=None, category=None, payload=None):
        pass

    def pop_range(domain=None):
        pass

    def start_range(message=None, color=None, domain=None, category=None, payload=None):
        pass

    def end_range(range_id):
        pass
