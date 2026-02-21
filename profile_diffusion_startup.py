#!/usr/bin/env python3
"""
Script to profile the startup time of sglang diffusion server.

Usage:
    python profile_diffusion_startup.py --model-path <path_to_model>

Example:
    python profile_diffusion_startup.py --model-path stabilityai/stable-diffusion-3.5-large

Environment variables:
    SGLANG_PROFILE_STARTUP: Set to "1" to enable profiling (default: "1")
    SGLANG_STARTUP_PROFILE_PATH: Path to save profiling JSON (optional)
"""

import argparse
import json
import os
import signal
import sys
import time

# Enable startup profiling by default
os.environ.setdefault("SGLANG_PROFILE_STARTUP", "1")


def main():
    parser = argparse.ArgumentParser(
        description="Profile sglang diffusion server startup time"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the diffusion model (local path or HuggingFace model ID)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save profiling JSON output",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port to bind the server (default: 30000)",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup requests to measure pure initialization time",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--dit-cpu-offload",
        action="store_true",
        help="Enable DiT CPU offload",
    )
    parser.add_argument(
        "--vae-cpu-offload",
        action="store_true",
        help="Enable VAE CPU offload",
    )
    parser.add_argument(
        "--text-encoder-cpu-offload",
        action="store_true",
        help="Enable text encoder CPU offload",
    )

    args = parser.parse_args()

    # Set output path if specified
    if args.output:
        os.environ["SGLANG_STARTUP_PROFILE_PATH"] = args.output

    print("=" * 60)
    print("SGLANG DIFFUSION SERVER STARTUP PROFILER")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Skip warmup: {args.skip_warmup}")
    print("=" * 60)

    start_time = time.perf_counter()

    # Import here to capture import time
    import_start = time.perf_counter()
    from sglang.multimodal_gen.runtime.launch_server import launch_server
    from sglang.multimodal_gen.runtime.server_args import prepare_server_args
    import_time = (time.perf_counter() - import_start) * 1000
    print(f"\nModule import time: {import_time:.2f} ms")

    # Build server args
    cli_args = [
        "--model-path", args.model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--num-gpus", str(args.num_gpus),
    ]

    if args.skip_warmup:
        cli_args.append("--no-warmup")

    if args.dit_cpu_offload:
        cli_args.append("--dit-cpu-offload")

    if args.vae_cpu_offload:
        cli_args.append("--vae-cpu-offload")

    if args.text_encoder_cpu_offload:
        cli_args.append("--text-encoder-cpu-offload")

    print(f"\nPreparing server args...")
    args_start = time.perf_counter()
    server_args = prepare_server_args(cli_args)
    args_time = (time.perf_counter() - args_start) * 1000
    print(f"Server args preparation time: {args_time:.2f} ms")

    print(f"\nLaunching server (press Ctrl+C to stop after initialization)...")

    # Set up signal handler to gracefully exit
    def signal_handler(signum, frame):
        total_time = (time.perf_counter() - start_time) * 1000
        print(f"\n\nTotal profiling time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Launch the server (this blocks until server is ready)
        # We don't launch HTTP server to just profile initialization
        processes = launch_server(server_args, launch_http_server=False)

        total_time = (time.perf_counter() - start_time) * 1000
        print(f"\n{'=' * 60}")
        print(f"TOTAL STARTUP TIME: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        print(f"{'=' * 60}")

        # Keep processes running briefly then exit
        print("\nServer initialized successfully. Shutting down...")
        for p in processes:
            p.terminate()
            p.join(timeout=5)

    except Exception as e:
        print(f"\nError during startup: {e}")
        raise


if __name__ == "__main__":
    main()
