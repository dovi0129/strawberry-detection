import os
import yaml
import cv2
import numpy as np
from ultralytics import YOLO

# 루트 기준 경로 계산
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg_path = os.path.join(project_root, 'configs/infer.yaml')

cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))

model = YOLO(os.path.join(project_root, cfg['weights']))

results = model.predict(
    source        = os.path.join(project_root, cfg['source']),
    save          = False,
    conf          = cfg.get('conf', 0.25),
    iou           = cfg.get('iou', 0.45),
    retina_masks  = True,
    overlap_mask  = False,
    batch         = cfg.get('batch', 16),
    imgsz         = cfg.get('imgsz', 640)
)

# 출력 폴더 구성
base      = os.path.join(project_root, 'runs', cfg.get('name', 'segment_exp_test'))
inst_f    = os.path.join(base, 'instance_masks')
comb_f    = os.path.join(base, 'combined_masks')
overlay_f = os.path.join(base, 'overlay_masks')
os.makedirs(inst_f,    exist_ok=True)
os.makedirs(comb_f,    exist_ok=True)
os.makedirs(overlay_f, exist_ok=True)

# 결과 처리 및 저장
for res in results:
    img_path = res.path
    img      = cv2.imread(img_path)
    masks    = res.masks.data.cpu().numpy()
    key      = os.path.splitext(os.path.basename(img_path))[0]

    for i, m in enumerate(masks):
        mask_png = (m * 255).astype('uint8')
        cv2.imwrite(f"{inst_f}/{key}_{i:02d}.png", mask_png)

    combined = np.any(masks, axis=0).astype('uint8') * 255
    cv2.imwrite(f"{comb_f}/{key}_combined.png", combined)

    colored = cv2.applyColorMap(cv2.convertScaleAbs(combined, alpha=0.5), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, colored, 0.3, 0)
    cv2.imwrite(f"{overlay_f}/{key}_overlay.png", overlay)

print("Instance masks  →", inst_f)
print("Combined masks  →", comb_f)
print("Overlay images  →", overlay_f)
