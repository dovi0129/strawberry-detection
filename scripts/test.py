import os
import yaml
import argparse
import io
import sys
import contextlib
from ultralytics import YOLO

# 프로젝트 최상위 디렉터리로 워킹 디렉터리 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/infer.yaml')
    args = parser.parse_args()

    # 설정 파일 경로
    cfg_path = os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model_path = os.path.join(PROJECT_ROOT, cfg['weights'])
    model = YOLO(model_path)

    # stdout을 버퍼로 리디렉션하여 내부 로그 숨기기
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        metrics = model.val(
            data     = os.path.join(PROJECT_ROOT, cfg['data']),
            split    = cfg.get('split', 'test'),
            imgsz    = cfg.get('imgsz', 640),
            batch    = cfg.get('batch', 16),
            project  = os.path.join(PROJECT_ROOT, cfg.get('project', 'runs')),
            name     = cfg.get('name', 'segment_exp_test'),
            exist_ok = True,
            verbose  = False,   # 단계별 상세 로그 꺼두기
            conf     = cfg.get('conf', 0.25),
            iou      = cfg.get('iou', 0.45)
        )
    # buf.getvalue()를 호출하지 않으면, 위 with 블록 안에서 출력된 내용은 화면에 전혀 보이지 않습니다.

    # 최종 지표만 깔끔하게 출력
    print('\n===== Test Metrics =====')
    # 필요하다면 metrics 전체 출력 대신, 원하는 컬럼만 선별해서 보여줄 수도 있습니다.
    print(metrics)

if __name__ == '__main__':
    main()
