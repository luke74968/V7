import os

# 적용할 새로운 라이선스 헤더
LICENSE_HEADER = """# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
"""

TARGET_DIRS = ["transformer_solver", "or_tools_solver", "common"]

def update_header_in_files():
    current_dir = os.getcwd()
    
    for target_dir in TARGET_DIRS:
        target_path = os.path.join(current_dir, target_dir)
        
        if not os.path.exists(target_path):
            continue

        print(f"📂 처리 중: {target_dir}...")
        
        for root, _, files in os.walk(target_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        
                        # --- 1. 헤더 영역(주석/빈줄)과 코드 영역 분리 ---
                        header_block = []
                        body_start_index = 0
                        
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            # 주석(#)으로 시작하거나 빈 줄이면 헤더 영역으로 간주
                            if stripped.startswith("#") or not stripped:
                                header_block.append(line)
                            else:
                                # 코드가 시작되면 중단
                                body_start_index = i
                                break
                        
                        body_block = lines[body_start_index:]
                        
                        # --- 2. 헤더 영역에서 '살려야 할 주석' 선별 ---
                        preserved_header = []
                        for line in header_block:
                            stripped = line.strip()
                            # [보호 규칙]
                            # 1. 셔뱅 (#!...)
                            # 2. 인코딩 (coding: utf-8 ...)
                            # 3. 파일명 주석 (.py로 끝나는 주석)
                            is_shebang = stripped.startswith("#!")
                            is_encoding = "coding:" in stripped or "-*-" in stripped
                            is_filepath = stripped.endswith(".py") and not "Copyright" in stripped
                            
                            if is_shebang or is_encoding or is_filepath:
                                preserved_header.append(line)
                                
                        # --- 3. 파일 다시 쓰기 ---
                        with open(file_path, "w", encoding="utf-8") as f:
                            # (1) 살려둔 주석 (파일명 등) 먼저 기록
                            for line in preserved_header:
                                f.write(line)
                            
                            # (2) 라이선스 헤더 추가
                            f.write(LICENSE_HEADER)
                            
                            # (3) 본문 코드 기록
                            for line in body_block:
                                f.write(line)
                            
                        print(f"  - ✅ Smart Updated: {file}")
                        
                    except Exception as e:
                        print(f"  - ❌ Error: {file} ({e})")

if __name__ == "__main__":
    update_header_in_files()
    print("\n✨ 라이선스 헤더 업데이트 완료 (파일명/셔뱅 주석 보호 적용).")