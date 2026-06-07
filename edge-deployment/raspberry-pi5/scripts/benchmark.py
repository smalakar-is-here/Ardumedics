#!/usr/bin/env python3
"""
ArduMedics Benchmark — Performance Profiling on Raspberry Pi 5

Measures inference speed, memory usage, and temperature for different model formats.

Usage:
    python3 benchmark.py --model models/best_nano_ncnn/
    python3 benchmark.py --model models/best_nano.pt --format pytorch
"""

import argparse
import time
import sys
import os
import subprocess

import cv2
import numpy as np


def get_pi_temperature():
    """Get Raspberry Pi CPU temperature."""
    try:
        result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "N/A"


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0


def benchmark_ncnn(model_dir, num_iterations=100, imgsz=640):
    """Benchmark NCNN model inference."""
    import ncnn

    param_path = os.path.join(model_dir, 'model.ncnn.param')
    bin_path = os.path.join(model_dir, 'model.ncnn.bin')

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    # Create dummy input
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    mat = ncnn.Mat.from_numpy(dummy.astype(np.float32))
    mat = (mat - 128.0) / 128.0

    # Warmup
    for _ in range(10):
        ex = net.create_extractor()
        ex.input("in0", mat)
        ex.extract("out0")

    # Benchmark
    times = []
    mem_before = get_memory_usage()

    for i in range(num_iterations):
        start = time.perf_counter()
        ex = net.create_extractor()
        ex.input("in0", mat)
        ex.extract("out0")
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mem_after = get_memory_usage()

    return {
        'format': 'NCNN',
        'times': times,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000.0 / np.mean(times),
        'memory_mb': mem_after - mem_before,
        'temperature': get_pi_temperature()
    }


def benchmark_pytorch(model_path, num_iterations=50, imgsz=640):
    """Benchmark PyTorch model inference."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    # Create dummy input
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model(dummy, verbose=False, imgsz=imgsz)

    # Benchmark
    times = []
    mem_before = get_memory_usage()

    for i in range(num_iterations):
        start = time.perf_counter()
        model(dummy, verbose=False, imgsz=imgsz)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    mem_after = get_memory_usage()

    return {
        'format': 'PyTorch',
        'times': times,
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000.0 / np.mean(times),
        'memory_mb': mem_after - mem_before,
        'temperature': get_pi_temperature()
    }


def main():
    parser = argparse.ArgumentParser(description='ArduMedics Pi 5 Benchmark')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--format', type=str, default='ncnn', choices=['ncnn', 'pytorch'],
                       help='Model format')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')

    args = parser.parse_args()

    print("=" * 60)
    print("  ArduMedics — Raspberry Pi 5 Benchmark")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Format: {args.format}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Temperature: {get_pi_temperature()}")
    print("=" * 60)

    if args.format == 'ncnn':
        result = benchmark_ncnn(args.model, args.iterations, args.imgsz)
    else:
        result = benchmark_pytorch(args.model, min(args.iterations, 50), args.imgsz)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — {result['format']}")
    print(f"{'=' * 60}")
    print(f"  Mean inference:  {result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms")
    print(f"  Min inference:   {result['min_ms']:.1f} ms")
    print(f"  Max inference:   {result['max_ms']:.1f} ms")
    print(f"  FPS:             {result['fps']:.1f}")
    print(f"  Memory delta:    {result['memory_mb']:.1f} MB")
    print(f"  Temperature:     {result['temperature']}")
    print(f"{'=' * 60}")

    # Save results
    import json
    output = {
        'model': args.model,
        'format': args.format,
        'imgsz': args.imgsz,
        'iterations': args.iterations,
        'mean_ms': round(result['mean_ms'], 1),
        'std_ms': round(result['std_ms'], 1),
        'fps': round(result['fps'], 1),
        'memory_mb': round(result['memory_mb'], 1),
        'temperature': result['temperature']
    }

    output_path = '/tmp/ardumedics_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
