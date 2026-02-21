# SPDX-License-Identifier: Apache-2.0
"""
Startup profiler for diffusion server.
Records timing for each startup phase.
"""

import atexit
import json
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class TimingRecord:
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    children: List["TimingRecord"] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class StartupProfiler:
    """
    A singleton profiler that records timing for each startup phase.
    Thread-safe and process-aware.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if StartupProfiler._initialized:
            return
        StartupProfiler._initialized = True

        self._records: OrderedDict[str, TimingRecord] = OrderedDict()
        self._stack: List[str] = []
        self._start_time = time.perf_counter()
        self._rank: Optional[int] = None
        self._enabled = os.environ.get("SGLANG_PROFILE_STARTUP", "1") == "1"
        self._output_path = os.environ.get("SGLANG_STARTUP_PROFILE_PATH", None)

        if self._enabled:
            atexit.register(self._on_exit)

    def set_rank(self, rank: int):
        """Set the current process rank."""
        self._rank = rank

    def _get_full_name(self, name: str) -> str:
        """Get the full hierarchical name."""
        if self._stack:
            return f"{self._stack[-1]}/{name}"
        return name

    def start(self, name: str, metadata: Optional[Dict] = None) -> str:
        """Start timing a phase. Returns the full name for later reference."""
        if not self._enabled:
            return name

        full_name = self._get_full_name(name)
        record = TimingRecord(
            name=full_name,
            start_time=time.perf_counter(),
            metadata=metadata or {},
        )

        # Add to parent's children if we have a parent
        if self._stack and self._stack[-1] in self._records:
            self._records[self._stack[-1]].children.append(record)

        self._records[full_name] = record
        self._stack.append(full_name)

        return full_name

    def end(self, name: Optional[str] = None):
        """End timing a phase."""
        if not self._enabled:
            return

        if name is None and self._stack:
            name = self._stack[-1]

        if name and name in self._records:
            record = self._records[name]
            record.end_time = time.perf_counter()
            record.duration_ms = (record.end_time - record.start_time) * 1000

            if self._stack and self._stack[-1] == name:
                self._stack.pop()

    @contextmanager
    def profile(self, name: str, metadata: Optional[Dict] = None):
        """Context manager for profiling a phase."""
        full_name = self.start(name, metadata)
        try:
            yield
        finally:
            self.end(full_name)

    def record(self, name: str, duration_ms: float, metadata: Optional[Dict] = None):
        """Directly record a timing without start/end."""
        if not self._enabled:
            return

        full_name = self._get_full_name(name)
        now = time.perf_counter()
        record = TimingRecord(
            name=full_name,
            start_time=now - duration_ms / 1000,
            end_time=now,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._records[full_name] = record

    def get_summary(self) -> Dict:
        """Get a summary of all timing records."""
        total_time = (time.perf_counter() - self._start_time) * 1000

        summary = {
            "rank": self._rank,
            "total_startup_time_ms": total_time,
            "phases": [],
        }

        # Get top-level records (not children)
        top_level_names = set()
        for name, record in self._records.items():
            if "/" not in name:
                top_level_names.add(name)

        for name in top_level_names:
            if name in self._records:
                record = self._records[name]
                phase_info = self._record_to_dict(record)
                summary["phases"].append(phase_info)

        # Sort by start time
        summary["phases"].sort(key=lambda x: x.get("start_offset_ms", 0))

        return summary

    def _record_to_dict(self, record: TimingRecord) -> Dict:
        """Convert a TimingRecord to a dictionary."""
        result = {
            "name": record.name,
            "duration_ms": record.duration_ms,
            "start_offset_ms": (record.start_time - self._start_time) * 1000,
        }

        if record.metadata:
            result["metadata"] = record.metadata

        if record.children:
            result["children"] = [
                self._record_to_dict(child) for child in record.children
            ]

        return result

    def print_summary(self):
        """Print a formatted summary to the logger."""
        if not self._enabled:
            return

        summary = self.get_summary()
        total_time = summary["total_startup_time_ms"]

        rank_str = f"[Rank {self._rank}] " if self._rank is not None else ""

        logger.info(f"\n{'=' * 60}")
        logger.info(f"{rank_str}STARTUP PROFILING SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total startup time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        logger.info(f"{'-' * 60}")

        def print_phase(phase: Dict, indent: int = 0):
            prefix = "  " * indent
            name = phase["name"].split("/")[-1]  # Get last part of hierarchical name
            duration = phase.get("duration_ms")
            if duration is not None:
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                logger.info(
                    f"{prefix}{name}: {duration:.2f} ms ({percentage:.1f}%)"
                )

                # Print children
                for child in phase.get("children", []):
                    print_phase(child, indent + 1)

        for phase in summary["phases"]:
            print_phase(phase)

        logger.info(f"{'=' * 60}\n")

    def save_to_file(self, path: Optional[str] = None):
        """Save the profiling summary to a JSON file."""
        if not self._enabled:
            return

        output_path = path or self._output_path
        if output_path is None:
            # Generate default path
            rank_suffix = f"_rank{self._rank}" if self._rank is not None else ""
            output_path = f"startup_profile{rank_suffix}.json"

        summary = self.get_summary()

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Startup profile saved to: {output_path}")

    def _on_exit(self):
        """Called when the process exits."""
        if self._rank == 0 or self._rank is None:
            self.print_summary()
            if self._output_path:
                self.save_to_file()


# Global instance
_profiler: Optional[StartupProfiler] = None


def get_startup_profiler() -> StartupProfiler:
    """Get the global startup profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = StartupProfiler()
    return _profiler


def profile_startup(name: str, metadata: Optional[Dict] = None):
    """Decorator for profiling a function's startup time."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_startup_profiler()
            with profiler.profile(name, metadata):
                return func(*args, **kwargs)

        return wrapper

    return decorator
