import os
import yaml
import cv2
import numpy as np
from ultralytics import YOLO

# 1) 설정 로드
cfg = yaml.safe_load(open('configs/infer.yaml', encoding='utf-8'))

# 2) 모델 로드
model = YOLO(cfg['weights'])

# 3) predict() → mask 저장용 결과 얻기
results = model.predict(
    source        = cfg['source'],      # 예: 'data/images/test'
    save          = False,
    conf          = cfg.get('conf', 0.25),
    iou           = cfg.get('iou', 0.45),
    retina_masks  = True,
    overlap_mask  = False,
    batch         = cfg.get('batch', 16),
    imgsz         = cfg.get('imgsz', 640)
)

# 4) 폴더 준비
base      = os.path.join('runs', cfg.get('name', 'segment_exp_test'))
inst_f    = os.path.join(base, 'instance_masks')
comb_f    = os.path.join(base, 'combined_masks')
overlay_f = os.path.join(base, 'overlay_masks')
os.makedirs(inst_f,    exist_ok=True)
os.makedirs(comb_f,    exist_ok=True)
os.makedirs(overlay_f, exist_ok=True)

# 5) 마스크 저장 & 오버레이 생성
for res in results:
    # (a) 원본 이미지 로드
    img_path = res.path
    img      = cv2.imread(img_path)

    # (b) 마스크 텐서를 NumPy로 변환 (N_instances, H, W)
    masks = res.masks.data.cpu().numpy()

    # (c) 이미지 파일명 키
    key = os.path.splitext(os.path.basename(img_path))[0]

    # --- 인스턴스별 마스크 저장 ---
    for i, m in enumerate(masks):
        mask_png = (m * 255).astype('uint8')
        cv2.imwrite(f"{inst_f}/{key}_{i:02d}.png", mask_png)

    # --- 통합 마스크 저장 ---
    combined = np.any(masks, axis=0).astype('uint8') * 255
    cv2.imwrite(f"{comb_f}/{key}_combined.png", combined)

    # --- 오버레이 생성: 컬러맵 & 반투명 합성 ---
    # 1) combined mask에 컬러맵 적용
    colored = cv2.applyColorMap(
        cv2.convertScaleAbs(combined, alpha=0.5),
        cv2.COLORMAP_JET
    )
    # 2) 원본 이미지와 컬러맵 마스크를 반투명으로 블렌딩
    overlay = cv2.addWeighted(img, 0.7, colored, 0.3, 0)

    # 3) 저장
    cv2.imwrite(f"{overlay_f}/{key}_overlay.png", overlay)

print("Instance masks  →", inst_f)
print("Combined masks  →", comb_f)
print("Overlay images  →", overlay_f)
