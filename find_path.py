# check_yolo_load.py
import os
from ultralytics import YOLO

# 1) 모델 경로 지정 (절대경로 권장)
model_path = os.path.abspath("models/weights/yolov8n-seg.pt")
print("▶ 로드할 모델 파일:", model_path, "→ 존재 여부:", os.path.exists(model_path))

# 2) 모델 로드 시도
model = YOLO(model_path)

# 3) 로드된 모델 정보 출력
print("▶ 모델 클래스:", type(model))
print("▶ 모델 체크포인트 확인:", model.ckpt)  # ckpt 에서 로드된 파일 경로가 보입니다.
