#!/usr/bin/env python3
"""
scripts/export.py

학습된 YOLOv8 세그멘테이션 모델(best.pt)을 ONNX와/또는 TensorRT 엔진(.engine) 포맷으로
순차적으로 내보내는 스크립트입니다.
"""

import argparse
import os
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLOv8 segmentation model to ONNX / TensorRT engine")
    parser.add_argument(
        '-w', '--weights', type=str,
        default=os.path.join('..','runs','segment_exp','weights','best.pt'),
        help='학습된 .pt 파일 경로')
    parser.add_argument(
        '-f', '--formats', nargs='+', default=['onnx'],
        choices=['onnx', 'engine'],
        help='내보낼 포맷들 (예: --formats onnx engine)')
    parser.add_argument(
        '-i', '--imgsz', nargs=2, type=int, default=[640,640],
        help='모델 입력 해상도 (width height)')
    parser.add_argument(
        '-b', '--batch-size', type=int, default=1,
        help='배치 사이즈 (Jetson 메모리에 맞춰 1 권장)')
    parser.add_argument(
        '-d', '--device', type=str, default='cpu',
        help='export 수행 디바이스 (cpu 또는 cuda)')
    parser.add_argument(
        '--simplify', action='store_true',
        help='ONNX 모델 단순화 (onnx-simplifier)')
    parser.add_argument(
        '--opset', type=int, default=12,
        help='ONNX opset 버전')
    parser.add_argument(
        '--dynamic', action='store_true',
        help='ONNX dynamic axes 활성화 (배치/해상도 동적)')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"[INFO] Loading model from `{args.weights}`")
    model = YOLO(args.weights)

    common_kwargs = {
        "imgsz": tuple(args.imgsz),
        "batch": args.batch_size,
        "device": args.device,
        "simplify": args.simplify,
        "opset": args.opset,
        "dynamic": args.dynamic
    }

    for fmt in args.formats:
        print(f"[INFO] Exporting to {fmt} ...")
        model.export(format=fmt, **common_kwargs)
        print(f"[INFO] → finished {fmt}")

    print("[INFO] All exports done.")

if __name__ == "__main__":
    main()
