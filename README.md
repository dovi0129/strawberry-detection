# Strawberry Detection & Pinch-Point Calculation

캡스톤 디자인 프로젝트
YOLOv8-seg 기반 딸기 객체 탐지 및 파지점(pinch point) 계산

---

## 프로젝트 개요

본 리포지토리는 캡스톤 디자인의 일환으로, Ultralytics의 YOLOv8-seg 모델을 활용하여

* 온실 이미지에서 딸기 객체를 검출
* 검출된 딸기의 최적 파지점(줄기 부위) 좌표를 계산
* 로봇 팔/수확기 연동을 위한 좌표 정보 출력

두 가지 핵심 모듈으로 구성되어 있습니다.

1. **객체 탐지 & 분할 (segmentation)**
2. **파지점(pinch-point) 계산** : 현재 해당 부분은 yolov8_test.ipynb 에만 구현

---

### 1. 데이터 준비

* `data/images/` 디렉토리에 원본 이미지(`.PNG` 등)
* `data/labels/` 에 YOLO 형식(bbox + class) 라벨링 파일(.txt)
* `configs/data.yaml` 에 데이터셋 경로 및 클래스 정보 설정

### 2. 모델 훈련

```bash
# 예시: configs/train.yaml 설정 참조
python scripts/train.py --config configs/train.yaml
```

* 훈련 중간체크포인트는 `runs/segment_exp/weights/` 폴더에 저장
* 학습 로그 및 시각화는 `runs/segment_exp/` 확인

### 3. 검출 & 파지점 계산

```bash
python scripts/test.py --config configs/infer.yaml
```

* 결과 이미지(`val_batch*_pred.jpg`) 및 성능 지표(`MaskPR_curve.png`, `confusion_matrix.png` 등)
  → `runs/segment_exp_test/` 에 저장
* 파지점 좌표는 콘솔 출력 또는 지정된 CSV 파일로 저장

---

## 주요 스크립트

* **`scripts/train.py`**
  └─ `--config` (`configs/train.yaml`) 파일을 읽어 YOLOv8-seg로 모델 학습
* **`scripts/infer.py`**
  └─ `--config` (`configs/infer.yaml`) 파일을 읽어 모델 평가
* **`notebooks/yolov8_test.ipynb`**
  └─ 노트북 환경에서 추론 결과 확인 및 수확 객체 판별 및 파지점 확인 가능

