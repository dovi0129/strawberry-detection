# train.py
import os
import yaml
import argparse
from ultralytics import YOLO
import shutil

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 세그멘테이션 모델 학습 스크립트"
    )
    parser.add_argument(
        '--config', '-c',
        default='configs/train.yaml',
        help='학습 설정 파일 경로 (YAML)'
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    cfg_path = os.path.join(base_dir, args.config)
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    rel_model = cfg.pop('model')
    model_path = os.path.join(base_dir, rel_model)

    model = YOLO(model_path)
    model.train(**cfg)
    shutil.move('yolo11n.pt',      'models/weights/yolo11n.pt')
    
if __name__ == '__main__':
    main()
