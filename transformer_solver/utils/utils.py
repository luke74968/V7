# transformer_solver/utils/utils.py

import os
import sys
import shutil

def copy_all_src(dst_root: str):
    """
    훈련 실행 시, 현재 실행된 소스 코드를 
    지정된 디렉토리(dst_root) 하위의 'src_backup' 폴더에 백업합니다.
    
    Args:
        dst_root (str): 백업을 저장할 루트 디렉토리 (예: 'transformer_solver/result/...')
    """
    try:
        # 현재 실행 중인 스크립트의 디렉토리를 기준으로 소스 경로를 찾습니다.
        execution_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        dst_path = os.path.join(dst_root, 'src_backup')

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # 실행 경로 하위의 모든 .py 파일을 탐색
        for root, _, files in os.walk(execution_path):
            for file in files:
                if file.endswith('.py'):
                    src_file_path = os.path.join(root, file)
                    
                    # 가상환경이나 설치된 패키지 폴더는 백업에서 제외
                    if 'site-packages' in src_file_path or \
                       'venv' in src_file_path or \
                       '.venv' in src_file_path:
                        continue

                    # 백업 대상 경로 계산
                    relative_path = os.path.relpath(src_file_path, execution_path)
                    dst_file_path = os.path.join(dst_path, relative_path)

                    # 대상 디렉토리 생성 및 파일 복사
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                    shutil.copy(src_file_path, dst_file_path)
                    
        print(f"Source code backed up to: {dst_path}")
        
    except Exception as e:
        # (백업 실패가 메인 훈련에 영향을 주지 않도록 예외 처리)
        print(f"Warning: Could not back up source code: {e}")