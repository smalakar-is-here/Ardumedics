#!/usr/bin/env python3
"""
ArduMedics — NCNN Model Export Script

Exports YOLOv8n-Pose (best_nano.pt) to NCNN format for Raspberry Pi 5.

Usage:
    python3 export_ncnn.py                          # Uses default paths
    python3 export_ncnn.py --model best_nano.pt     # Custom model path
    python3 export_ncnn.py --model best_nano.pt --output best_nano_ncnn/

Requirements:
    pip install ultralytics
"""

import os
import sys
import argparse
from pathlib import Path


def export_ncnn(model_path, output_dir=None):
    """Export YOLOv8-Pose model to NCNN format."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please download best_nano.pt from Kaggle NB01 output first.")
        sys.exit(1)

    print(f"[ArduMedics] Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"[ArduMedics] Exporting to NCNN format...")
    export_path = model.export(format='ncnn')

    # The export creates a 'best_nano_ncnn_model/' folder next to the .pt file
    exported_folder = Path(export_path)

    if output_dir and str(exported_folder) != output_dir:
        import shutil
        target = Path(output_dir)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(exported_folder, target)
        print(f"[ArduMedics] NCNN model copied to: {target}")
    else:
        print(f"[ArduMedics] NCNN model exported to: {exported_folder}")

    # Verify
    ncnn_dir = Path(output_dir) if output_dir else exported_folder
    param_file = ncnn_dir / "model.ncnn.param"
    bin_file = ncnn_dir / "model.ncnn.bin"

    if param_file.exists() and bin_file.exists():
        param_size = param_file.stat().st_size
        bin_size = bin_file.stat().st_size
        print(f"\n[ArduMedics] Export successful!")
        print(f"  model.ncnn.param: {param_size / 1024:.1f} KB")
        print(f"  model.ncnn.bin:   {bin_size / (1024*1024):.1f} MB")
        print(f"\n  Ready for Raspberry Pi 5 deployment!")
        print(f"  Usage: python3 run_fall_detection.py --model {ncnn_dir}/ --camera 0")
    else:
        print(f"\n[ERROR] NCNN export files not found in {ncnn_dir}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLOv8n-Pose to NCNN for Pi 5')
    parser.add_argument('--model', type=str, default='best_nano.pt',
                       help='Path to best_nano.pt (default: best_nano.pt)')
    parser.add_argument('--output', type=str, default='best_nano_ncnn',
                       help='Output directory name (default: best_nano_ncnn)')

    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model
    output_dir = script_dir / args.output

    export_ncnn(str(model_path), str(output_dir))
