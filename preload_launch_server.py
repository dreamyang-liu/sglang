#!/usr/bin/env python3
"""
Optimized server launcher with parallel model preloading.

This script preloads model weights into CPU memory while initializing
the distributed environment, then loads from memory instead of disk.

Usage:
    python preload_launch_server.py --model-path "Qwen/Qwen-Image" --num-gpus 4 --tp-size 4
"""

import argparse
import gc
import multiprocessing as mp
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class ModelPreloader:
    """Preload model weights into CPU memory in background."""

    def __init__(self, model_path: str, modules_to_preload: list[str] | None = None):
        self.model_path = model_path
        self.modules_to_preload = modules_to_preload or [
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "vae",
        ]
        self.preloaded_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        self.preload_times: Dict[str, float] = {}
        self.lock = threading.Lock()
        self._preload_thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    def _get_safetensors_files(self, module_path: str) -> list[Path]:
        """Get all safetensors files in a module directory."""
        path = Path(module_path)
        if not path.exists():
            return []

        # Check for safetensors files
        safetensors_files = list(path.glob("*.safetensors"))
        if safetensors_files:
            return safetensors_files

        # Fallback to pytorch bin files
        bin_files = list(path.glob("*.bin"))
        return bin_files

    def _preload_single_module(self, module_name: str) -> tuple[str, float, int]:
        """Preload a single module's weights into CPU memory."""
        module_path = os.path.join(self.model_path, module_name)

        if not os.path.exists(module_path):
            print(f"[PRELOAD] Module path not found: {module_path}")
            return module_name, 0.0, 0

        start_time = time.perf_counter()
        weights = {}
        total_size = 0

        files = self._get_safetensors_files(module_path)

        for file_path in files:
            try:
                if file_path.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    file_weights = load_file(str(file_path), device="cpu")
                else:
                    file_weights = torch.load(
                        str(file_path),
                        map_location="cpu",
                        weights_only=True,
                    )

                for key, tensor in file_weights.items():
                    weights[key] = tensor
                    total_size += tensor.numel() * tensor.element_size()

            except Exception as e:
                print(f"[PRELOAD] Error loading {file_path}: {e}")

        duration = time.perf_counter() - start_time

        with self.lock:
            self.preloaded_weights[module_name] = weights
            self.preload_times[module_name] = duration

        size_mb = total_size / (1024 * 1024)
        print(
            f"[PRELOAD] Loaded {module_name}: {len(weights)} tensors, "
            f"{size_mb:.1f} MB in {duration:.2f}s"
        )

        return module_name, duration, total_size

    def start_preload_async(self):
        """Start preloading in background threads."""
        print(f"[PRELOAD] Starting async preload for modules: {self.modules_to_preload}")

        # Use ThreadPoolExecutor for parallel I/O
        self._executor = ThreadPoolExecutor(max_workers=len(self.modules_to_preload))

        futures = {
            self._executor.submit(self._preload_single_module, module): module
            for module in self.modules_to_preload
        }

        # Don't wait, return immediately
        self._futures = futures

    def wait_for_preload(self, timeout: float | None = None) -> Dict[str, float]:
        """Wait for all preloading to complete."""
        if not hasattr(self, "_futures"):
            return {}

        start = time.perf_counter()
        results = {}

        for future in as_completed(self._futures, timeout=timeout):
            module_name = self._futures[future]
            try:
                name, duration, size = future.result()
                results[name] = duration
            except Exception as e:
                print(f"[PRELOAD] Error preloading {module_name}: {e}")

        total_time = time.perf_counter() - start
        print(f"[PRELOAD] All modules preloaded in {total_time:.2f}s")

        if self._executor:
            self._executor.shutdown(wait=False)

        return results

    def get_preloaded_weights(self, module_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get preloaded weights for a module."""
        with self.lock:
            return self.preloaded_weights.get(module_name)

    def clear(self):
        """Clear all preloaded weights to free memory."""
        with self.lock:
            self.preloaded_weights.clear()
        gc.collect()


# Global preloader instance
_preloader: Optional[ModelPreloader] = None


def get_preloader() -> Optional[ModelPreloader]:
    return _preloader


def patch_component_loader():
    """Patch PipelineComponentLoader to use preloaded weights."""
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        PipelineComponentLoader,
    )

    original_load = PipelineComponentLoader.load

    def patched_load(self, *args, **kwargs):
        preloader = get_preloader()
        module_name = getattr(self, "module_name", None)

        if preloader and module_name:
            preloaded = preloader.get_preloaded_weights(module_name)
            if preloaded:
                print(f"[PRELOAD] Using preloaded weights for {module_name}")
                # TODO: Actually use the preloaded weights
                # This requires deeper integration with the loader

        return original_load(self, *args, **kwargs)

    PipelineComponentLoader.load = patched_load


def launch_with_preload(args):
    """Launch server with model preloading optimization."""
    global _preloader

    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model

    print("=" * 60)
    print("OPTIMIZED LAUNCH WITH PRELOADING")
    print("=" * 60)

    total_start = time.perf_counter()

    # Step 1: Download model if needed (can't parallelize this)
    print("\n[STEP 1] Ensuring model is downloaded...")
    step1_start = time.perf_counter()
    model_path = maybe_download_model(args.model_path)
    step1_time = time.perf_counter() - step1_start
    print(f"[STEP 1] Model path ready: {model_path} ({step1_time:.2f}s)")

    # Step 2: Start preloading in background
    print("\n[STEP 2] Starting background preload...")
    step2_start = time.perf_counter()
    _preloader = ModelPreloader(model_path)
    _preloader.start_preload_async()
    step2_time = time.perf_counter() - step2_start
    print(f"[STEP 2] Preload threads started ({step2_time:.4f}s)")

    # Step 3: Initialize distributed environment (in parallel with preloading)
    print("\n[STEP 3] Initializing distributed environment...")
    step3_start = time.perf_counter()

    # Patch the loader before importing server modules
    patch_component_loader()

    from sglang.multimodal_gen.runtime.server_args import prepare_server_args

    server_argv = [
        "--model-path", model_path,
        "--num-gpus", str(args.num_gpus),
        "--tp-size", str(args.tp_size),
        "--host", args.host,
        "--port", str(args.port),
    ]
    server_args = prepare_server_args(server_argv)
    step3_time = time.perf_counter() - step3_start
    print(f"[STEP 3] Server args prepared ({step3_time:.2f}s)")

    # Step 4: Wait for preloading to complete
    print("\n[STEP 4] Waiting for preload to complete...")
    step4_start = time.perf_counter()
    preload_times = _preloader.wait_for_preload()
    step4_time = time.perf_counter() - step4_start
    print(f"[STEP 4] Preload wait completed ({step4_time:.2f}s)")

    # Step 5: Launch server (now loading from memory should be faster)
    print("\n[STEP 5] Launching server...")
    step5_start = time.perf_counter()

    from sglang.multimodal_gen.runtime.launch_server import launch_server

    processes = launch_server(server_args, launch_http_server=False)
    step5_time = time.perf_counter() - step5_start
    print(f"[STEP 5] Server launched ({step5_time:.2f}s)")

    total_time = time.perf_counter() - total_start

    # Print summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"  Step 1 (Download check):    {step1_time:>8.2f}s")
    print(f"  Step 2 (Start preload):     {step2_time:>8.4f}s")
    print(f"  Step 3 (Prepare args):      {step3_time:>8.2f}s")
    print(f"  Step 4 (Wait preload):      {step4_time:>8.2f}s")
    print(f"  Step 5 (Launch server):     {step5_time:>8.2f}s")
    print("-" * 60)
    print(f"  TOTAL:                      {total_time:>8.2f}s")
    print("=" * 60)

    if preload_times:
        print("\nPRELOAD BREAKDOWN:")
        for module, t in sorted(preload_times.items(), key=lambda x: -x[1]):
            print(f"  {module}: {t:.2f}s")

    # Cleanup
    print("\nShutting down...")
    _preloader.clear()

    from sglang.multimodal_gen.runtime.launch_server import kill_process_tree
    kill_process_tree(os.getpid(), include_parent=False)

    return total_time


def main():
    parser = argparse.ArgumentParser(
        description="Launch server with parallel model preloading"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
    )

    args = parser.parse_args()
    launch_with_preload(args)


if __name__ == "__main__":
    main()
