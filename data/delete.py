import os
import shutil

# 삭제할 폴더 경로(상대경로 또는 절대경로)
folder_path = 'data'  

# 폴더가 존재하면 삭제
if os.path.isdir(folder_path):
    shutil.rmtree(folder_path)
    print(f'✅ 폴더를 삭제했습니다: {folder_path}')
else:
    print(f'⚠️ 폴더가 존재하지 않습니다: {folder_path}')
