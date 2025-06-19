import os
import shutil

runs_dir = '../runs'  # runs 폴더 경로 (필요에 따라 경로 수정)

# 1. runs 폴더 내 *_exp로 끝나는 폴더 전부 삭제
for fname in os.listdir(runs_dir):
    path = os.path.join(runs_dir, fname)
    # 폴더명 끝이 _exp 또는 _expN (예: segment_exp, detect_exp2 등) 인지 확인
    if os.path.isdir(path) and fname.endswith('_exp') or fname.endswith('_exp1') or fname.endswith('_exp2') or fname.endswith('_exp3') or fname.endswith('_exp4') or fname.endswith('ouput') or fname.endswith('_test') or fname.endswith('_test2') or fname.endswith('val') or fname.endswith('val2'):
        print(f"삭제: {path}")
        shutil.rmtree(path)

# 2. runs 폴더 및 하위 폴더 내 val 폴더 전부 삭제
for root, dirs, files in os.walk(runs_dir):
    for d in dirs:
        if d == 'val':
            val_path = os.path.join(root, d)
            print(f"삭제: {val_path}")
            shutil.rmtree(val_path)

print("YOLOv8 실험 및 val 폴더 삭제 완료!")
