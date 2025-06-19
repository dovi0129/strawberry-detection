import os
import yaml
import argparse
from ultralytics import YOLO
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

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

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg_path = os.path.join(project_root, args.config)
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    rel_model = cfg.pop('model')
    model_path = os.path.join(project_root, rel_model)

    model = YOLO(model_path)
    model.train(**cfg)

    # 훈련 후 생성된 모델 파일 이동
    output_path = os.path.join(project_root, 'models/weights/yolo11n.pt')
    shutil.move('yolo11n.pt', output_path)

if __name__ == '__main__':
    main()
