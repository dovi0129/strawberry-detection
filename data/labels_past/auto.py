import os

labels_dir = 'test'  # test 라벨 폴더 경로
for fname in os.listdir(labels_dir):
    if fname.endswith('.txt'):
        fpath = os.path.join(labels_dir, fname)
        new_lines = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # class, cx, cy, w, h = 5개, polygon 좌표 = 6개 이상(3쌍) 필요
                if len(parts) >= 11:  # 최소 5+6=11개 이상
                    new_lines.append(line)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
