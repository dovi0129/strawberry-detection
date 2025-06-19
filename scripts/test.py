import yaml, argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config','-c', default='configs/infer.yaml')
    args = p.parse_args()

    # 1) 설정 불러오기
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 2) 모델 로드
    model = YOLO(cfg['weights'])

    # 3) Test (metrics + 시각화 플롯)
    metrics = model.val(
        data     = cfg['data'],
        split    = cfg.get('split','test'),
        imgsz    = cfg.get('imgsz', 640),   # ← 반드시 추가
        batch    = cfg.get('batch',16),     # ← 반드시 추가
        project  = cfg.get('project','runs'),
        name     = cfg.get('name','segment_exp_test'),
        exist_ok = True,
        plots    = True,
        conf     = cfg.get('conf',0.25),
        iou      = cfg.get('iou',0.45),
        verbose  = True
    )

    # 4) 결과 출력
    print('\n===== Test Metrics =====')
    print(metrics)

if __name__ == '__main__':
    main()
