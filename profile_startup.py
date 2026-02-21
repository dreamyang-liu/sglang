#!/usr/bin/env python3
"""
Profile the startup time of sglang diffusion server.

This script runs initialization in the CURRENT process (not subprocess)
to capture detailed timing of each phase.

Usage:
    python profile_startup.py --model-path "Qwen/Qwen-Image" --num-gpus 1 --tp-size 1
"""

import argparse
import functools
import gc
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import torch


@dataclass
class TimingRecord:
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    children: list = field(default_factory=list)


class StartupProfiler:
    """Global profiler to track startup timing."""

    def __init__(self):
        self.records: list[TimingRecord] = []
        self.stack: list[TimingRecord] = []
        self.start_time = time.perf_counter()
        self.module_load_times: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure time for a code block."""
        record = TimingRecord(name=name, start_time=time.perf_counter())
        if self.stack:
            self.stack[-1].children.append(record)
        else:
            self.records.append(record)
        self.stack.append(record)

        try:
            yield record
        finally:
            record.end_time = time.perf_counter()
            record.duration_ms = (record.end_time - record.start_time) * 1000
            self.stack.pop()
            print(f"[PROFILE] {name}: {record.duration_ms:.2f} ms")

    def record_module_load(self, module_name: str, duration_ms: float):
        """Record time for loading a specific module."""
        self.module_load_times[module_name] = duration_ms

    def get_total_time_ms(self) -> float:
        """Get total elapsed time since profiler started."""
        return (time.perf_counter() - self.start_time) * 1000

    def print_summary(self):
        """Print a summary of all timing records."""
        print("\n" + "=" * 70)
        print("STARTUP PROFILE SUMMARY")
        print("=" * 70)

        def print_record(record: TimingRecord, indent: int = 0):
            prefix = "  " * indent
            pct = ""
            if indent == 0 and self.records:
                total = self.records[0].duration_ms
                if total > 0:
                    pct = f" ({record.duration_ms / total * 100:.1f}%)"
            print(f"{prefix}{record.name}: {record.duration_ms:.2f} ms{pct}")
            for child in record.children:
                print_record(child, indent + 1)

        for record in self.records:
            print_record(record)

        if self.module_load_times:
            print("\n" + "-" * 70)
            print("MODULE LOAD TIMES:")
            print("-" * 70)
            total_module_time = sum(self.module_load_times.values())
            for name, duration in sorted(
                self.module_load_times.items(), key=lambda x: -x[1]
            ):
                pct = duration / total_module_time * 100 if total_module_time > 0 else 0
                print(f"  {name}: {duration:.2f} ms ({pct:.1f}%)")
            print(f"  --- Total: {total_module_time:.2f} ms ---")

        print("\n" + "-" * 70)
        print(f"TOTAL STARTUP TIME: {self.get_total_time_ms():.2f} ms")
        print("=" * 70)

    def to_dict(self) -> dict:
        """Convert profiling results to dictionary."""

        def record_to_dict(record: TimingRecord) -> dict:
            return {
                "name": record.name,
                "duration_ms": record.duration_ms,
                "children": [record_to_dict(c) for c in record.children],
            }

        return {
            "total_time_ms": self.get_total_time_ms(),
            "phases": [record_to_dict(r) for r in self.records],
            "module_load_times": self.module_load_times,
        }


# Global profiler instance
profiler = StartupProfiler()


def run_profiled_init(args):
    """Run model initialization with detailed profiling."""

    print("=" * 70)
    print("DETAILED STARTUP PROFILING")
    print(f"Model: {args.model_path}")
    print(f"GPUs: {args.num_gpus}, TP: {args.tp_size}")
    print("=" * 70)

    with profiler.measure("total_init"):

        # Step 1: Import and prepare
        with profiler.measure("imports"):
            from sglang.multimodal_gen.runtime.server_args import ServerArgs
            from sglang.multimodal_gen.runtime.server_args import prepare_server_args

        # Step 2: Prepare server args (this also downloads model if needed)
        with profiler.measure("prepare_server_args"):
            server_argv = [
                "--model-path", args.model_path,  # Use original model name
                "--num-gpus", str(args.num_gpus),
                "--tp-size", str(args.tp_size),
                "--host", "127.0.0.1",
                "--port", str(args.port),
            ]
            server_args = prepare_server_args(server_argv)
            model_path = server_args.model_path
            print(f"  Model path: {model_path}")

        # Step 4: Set CUDA device
        with profiler.measure("set_cuda_device"):
            local_rank = 0
            torch.cuda.set_device(local_rank)
            print(f"  Using GPU: {torch.cuda.get_device_name(local_rank)}")

        # Step 5: Initialize distributed environment
        with profiler.measure("distributed_init"):
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(server_args.master_port)
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = str(args.num_gpus)

            from sglang.multimodal_gen.runtime.distributed import (
                maybe_init_distributed_environment_and_model_parallel,
            )

            maybe_init_distributed_environment_and_model_parallel(
                tp_size=server_args.tp_size,
                enable_cfg_parallel=server_args.enable_cfg_parallel,
                ulysses_degree=server_args.ulysses_degree,
                ring_degree=server_args.ring_degree,
                sp_size=server_args.sp_degree,
                dp_size=server_args.dp_size,
                distributed_init_method=f"tcp://127.0.0.1:{server_args.master_port}",
                dist_timeout=server_args.dist_timeout,
            )

        # Step 6: Build pipeline (this is the main part)
        with profiler.measure("build_pipeline"):

            # 6a: Get model info
            with profiler.measure("get_model_info"):
                from sglang.multimodal_gen.registry import get_model_info

                # Use original model path for get_model_info
                model_info = get_model_info(args.model_path, backend=server_args.backend)
                pipeline_cls = model_info.pipeline_cls
                print(f"  Pipeline class: {pipeline_cls.__name__}")

            # 6b: Instantiate pipeline (loads all modules)
            with profiler.measure("pipeline_instantiate"):
                # We need to profile individual module loads
                # Patch the loader before instantiation
                from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
                    PipelineComponentLoader,
                )

                original_load_component = PipelineComponentLoader.load_component

                @staticmethod
                def profiled_load_component(
                    component_name, component_model_path, transformers_or_diffusers, srv_args
                ):
                    start = time.perf_counter()
                    result = original_load_component(
                        component_name, component_model_path, transformers_or_diffusers, srv_args
                    )
                    duration_ms = (time.perf_counter() - start) * 1000
                    profiler.record_module_load(component_name, duration_ms)
                    return result

                PipelineComponentLoader.load_component = profiled_load_component

                # Now instantiate
                pipeline = pipeline_cls(model_path, server_args)

        # Step 7: Configure layerwise offload if enabled
        if server_args.dit_layerwise_offload:
            with profiler.measure("configure_layerwise_offload"):
                from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
                    OffloadableDiTMixin,
                )

                for dit in filter(
                    None,
                    [
                        pipeline.get_module("transformer"),
                        pipeline.get_module("transformer_2"),
                    ],
                ):
                    if isinstance(dit, OffloadableDiTMixin):
                        dit.configure_layerwise_offload(server_args)

    # Print summary
    profiler.print_summary()

    # Save results
    output_file = args.output or f"startup_profile_{args.model_path.replace('/', '_')}.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "num_gpus": args.num_gpus,
        "tp_size": args.tp_size,
        "profile": profiler.to_dict(),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PROFILE] Results saved to: {output_file}")

    # Cleanup
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return profiler


def main():
    parser = argparse.ArgumentParser(
        description="Profile sglang diffusion server startup time"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path (HuggingFace repo ID or local path)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (default: 1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port (default: 30000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for profiling results",
    )

    args = parser.parse_args()

    # Only support single GPU for this profiler (no subprocess)
    if args.num_gpus > 1:
        print("WARNING: This profiler only supports single GPU mode (num_gpus=1).")
        print("For multi-GPU profiling, the timing would be collected from rank 0 only.")
        print("Continuing with num_gpus=1 equivalent behavior...")

    run_profiled_init(args)


if __name__ == "__main__":
    main()
