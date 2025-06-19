# utils/trainer.py
import yaml
from ultralytics import YOLO

def run_train(cfg):
    # 1) 모델 초기화
    model = YOLO(cfg['model']['arch'])  # 예: 'yolov8n-seg.pt'
    # 2) 학습 실행
    model.train(
        data     = cfg['data'],
        task     = cfg.get('task', 'segment'),
        epochs   = cfg['train_args']['epochs'],
        imgsz    = cfg['train_args']['imgsz'],
        batch    = cfg['train_args']['batch'],
        augment  = cfg['train_args'].get('augment', True),
        project  = cfg['train_args'].get('project', 'runs'),
        name     = cfg['train_args'].get('name', 'exp')
    )
