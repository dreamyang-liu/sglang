#!/usr/bin/env python3
"""
Benchmark script for measuring model startup time across different GPU and TP configurations.

Usage:
    python benchmark_startup_time.py [--output results.json] [--timeout 600]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import requests


@dataclass
class BenchmarkConfig:
    model_path: str
    num_gpus: int
    tp_size: int


@dataclass
class BenchmarkResult:
    model_path: str
    num_gpus: int
    tp_size: int
    startup_time_seconds: Optional[float]
    success: bool
    error_message: Optional[str] = None
    run_index: int = 0  # Which run this is (0-indexed)


@dataclass
class AggregatedResult:
    model_path: str
    num_gpus: int
    tp_size: int
    num_runs: int
    successful_runs: int
    startup_times: list  # List of successful startup times
    avg_time: Optional[float] = None
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    std_time: Optional[float] = None


MODELS = [
    "black-forest-labs/FLUX.2-klein-4B",
    "zai-org/GLM-Image",
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "Qwen/Qwen-Image",
    "hunyuanvideo-community/HunyuanVideo",
]

# Valid configurations: gpu count and tp_size combinations
# Note: tp_size must be <= num_gpus and num_gpus must be divisible by tp_size
GPU_CONFIGS = [
    {"num_gpus": 1, "tp_size": 1},
    {"num_gpus": 2, "tp_size": 1},
    {"num_gpus": 2, "tp_size": 2},
    {"num_gpus": 4, "tp_size": 1},
    {"num_gpus": 4, "tp_size": 2},
    {"num_gpus": 4, "tp_size": 4},
]


def get_available_gpus() -> int:
    """Get the number of available GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len(result.stdout.strip().split("\n"))
    except Exception:
        pass
    return 0


def find_free_port(start_port: int = 30000) -> int:
    """Find a free port starting from start_port."""
    import socket

    port = start_port
    while port < 65535:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            port += 1
    raise RuntimeError("No free port found")


def wait_for_server(host: str, port: int, timeout: float = 600) -> tuple[bool, float]:
    """
    Wait for the server to become ready.

    Returns:
        (success, time_elapsed)
    """
    start_time = time.time()
    url = f"http://{host}:{port}/health"

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                return True, elapsed
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    return False, time.time() - start_time


def kill_process_tree(pid: int):
    """Kill a process and all its children."""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
    except Exception as e:
        print(f"Warning: Failed to kill process tree: {e}")
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def run_benchmark(
    model_path: str,
    num_gpus: int,
    tp_size: int,
    timeout: float = 600,
    port: Optional[int] = None,
    run_index: int = 0,
    total_runs: int = 1,
) -> BenchmarkResult:
    """
    Run a single benchmark for a model with specific GPU/TP configuration.

    Args:
        model_path: HuggingFace model path
        num_gpus: Number of GPUs to use
        tp_size: Tensor parallelism size
        timeout: Maximum time to wait for server startup (seconds)
        port: Port to use (auto-detected if None)
        run_index: Current run index (0-indexed)
        total_runs: Total number of runs for this configuration

    Returns:
        BenchmarkResult with timing and status information
    """
    if port is None:
        port = find_free_port()

    host = "127.0.0.1"

    # Build the command
    cmd = [
        sys.executable,
        "-m", "sglang.multimodal_gen.runtime.launch_server",
        "--model-path", model_path,
        "--num-gpus", str(num_gpus),
        "--tp-size", str(tp_size),
        "--host", host,
        "--port", str(port),
    ]

    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"GPUs: {num_gpus}, TP: {tp_size}, Run: {run_index + 1}/{total_runs}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    process = None
    try:
        # Start the server process
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for server to become ready
        success, elapsed = wait_for_server(host, port, timeout)

        if success:
            print(f"SUCCESS: Server started in {elapsed:.2f} seconds")
            return BenchmarkResult(
                model_path=model_path,
                num_gpus=num_gpus,
                tp_size=tp_size,
                startup_time_seconds=elapsed,
                success=True,
                run_index=run_index,
            )
        else:
            # Collect any error output
            error_output = ""
            if process.stdout:
                try:
                    process.stdout.close()
                except Exception:
                    pass

            print(f"FAILED: Server did not start within {timeout} seconds")
            return BenchmarkResult(
                model_path=model_path,
                num_gpus=num_gpus,
                tp_size=tp_size,
                startup_time_seconds=None,
                success=False,
                error_message=f"Timeout after {timeout} seconds",
                run_index=run_index,
            )

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return BenchmarkResult(
            model_path=model_path,
            num_gpus=num_gpus,
            tp_size=tp_size,
            startup_time_seconds=None,
            success=False,
            error_message=str(e),
            run_index=run_index,
        )
    finally:
        # Clean up the server process
        if process is not None:
            print("Shutting down server...")
            kill_process_tree(process.pid)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Warning: Process did not terminate gracefully")
            time.sleep(2)  # Give some time for ports to be released


def aggregate_results(results: list) -> list:
    """Aggregate results by model/gpu/tp configuration."""
    import statistics

    # Group by (model, num_gpus, tp_size)
    grouped = {}
    for r in results:
        key = (r["model_path"], r["num_gpus"], r["tp_size"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    aggregated = []
    for (model, num_gpus, tp_size), runs in grouped.items():
        successful_times = [
            r["startup_time_seconds"]
            for r in runs
            if r["success"] and r["startup_time_seconds"] is not None
        ]

        agg = {
            "model_path": model,
            "num_gpus": num_gpus,
            "tp_size": tp_size,
            "num_runs": len(runs),
            "successful_runs": len(successful_times),
            "startup_times": successful_times,
            "avg_time": None,
            "min_time": None,
            "max_time": None,
            "std_time": None,
        }

        if successful_times:
            agg["avg_time"] = statistics.mean(successful_times)
            agg["min_time"] = min(successful_times)
            agg["max_time"] = max(successful_times)
            if len(successful_times) > 1:
                agg["std_time"] = statistics.stdev(successful_times)

        aggregated.append(agg)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model startup time across different GPU/TP configurations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="startup_benchmark_results.json",
        help="Output file for results (JSON format)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Timeout in seconds for each server startup (default: 600)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to test (default: all models)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="Specific GPU counts to test (default: 1, 2, 4)",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=30000,
        help="Starting port number (default: 30000)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to repeat each test (default: 1)",
    )

    args = parser.parse_args()

    # Check available GPUs
    available_gpus = get_available_gpus()
    print(f"Available GPUs: {available_gpus}")

    if available_gpus == 0:
        print("ERROR: No GPUs detected!")
        sys.exit(1)

    # Filter models if specified
    models_to_test = args.models if args.models else MODELS

    # Filter GPU configurations based on available GPUs and user specification
    configs_to_test = []
    for config in GPU_CONFIGS:
        if config["num_gpus"] <= available_gpus:
            if args.gpus is None or config["num_gpus"] in args.gpus:
                configs_to_test.append(config)

    if not configs_to_test:
        print("ERROR: No valid GPU configurations to test!")
        sys.exit(1)

    print(f"\nModels to test: {models_to_test}")
    print(f"GPU configurations to test: {configs_to_test}")
    print(f"Repeat count: {args.repeat}")

    # Run benchmarks
    results = []
    total_tests = len(models_to_test) * len(configs_to_test) * args.repeat
    current_test = 0
    port = args.start_port

    for model in models_to_test:
        for config in configs_to_test:
            for run_idx in range(args.repeat):
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] Running benchmark...")

                result = run_benchmark(
                    model_path=model,
                    num_gpus=config["num_gpus"],
                    tp_size=config["tp_size"],
                    timeout=args.timeout,
                    port=port,
                    run_index=run_idx,
                    total_runs=args.repeat,
                )
                results.append(asdict(result))

                # Increment port for next test
                port += 100
                if port > 60000:
                    port = args.start_port

    # Aggregate results if repeat > 1
    aggregated = aggregate_results(results) if args.repeat > 1 else None

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "available_gpus": available_gpus,
        "timeout_seconds": args.timeout,
        "repeat_count": args.repeat,
        "results": results,
    }
    if aggregated:
        output_data["aggregated"] = aggregated

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output}")

    # Print summary table
    if args.repeat > 1 and aggregated:
        # Print aggregated results
        print("\n" + "="*100)
        print(f"{'Model':<35} {'GPUs':<6} {'TP':<4} {'Runs':<6} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Std (s)':<10}")
        print("="*100)

        for r in aggregated:
            model_short = r["model_path"].split("/")[-1][:32]
            avg_str = f"{r['avg_time']:.2f}" if r['avg_time'] else "N/A"
            min_str = f"{r['min_time']:.2f}" if r['min_time'] else "N/A"
            max_str = f"{r['max_time']:.2f}" if r['max_time'] else "N/A"
            std_str = f"{r['std_time']:.2f}" if r['std_time'] else "N/A"
            runs_str = f"{r['successful_runs']}/{r['num_runs']}"
            print(f"{model_short:<35} {r['num_gpus']:<6} {r['tp_size']:<4} {runs_str:<6} {avg_str:<10} {min_str:<10} {max_str:<10} {std_str:<10}")

        print("="*100)
    else:
        # Print individual results
        print("\n" + "="*80)
        print(f"{'Model':<45} {'GPUs':<6} {'TP':<4} {'Time (s)':<12} {'Status':<10}")
        print("="*80)

        for r in results:
            model_short = r["model_path"].split("/")[-1][:40]
            time_str = f"{r['startup_time_seconds']:.2f}" if r['startup_time_seconds'] else "N/A"
            status = "OK" if r["success"] else "FAILED"
            print(f"{model_short:<45} {r['num_gpus']:<6} {r['tp_size']:<4} {time_str:<12} {status:<10}")

        print("="*80)

    # Summary statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print(f"\nTotal runs: {len(results)}, Successful: {len(successful)}, Failed: {len(failed)}")


if __name__ == "__main__":
    main()
