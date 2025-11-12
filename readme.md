# 서버관련 셋업
# 가상환경 생성 (v6_env 라는 이름으로, Python 3.10 버전을 사용)
conda create -n v7_env python=3.12 -y
# 가상환경 활성화 
conda activate v7_env
# (가상환경이 (v7_env) 로 바뀐 것을 확인)
# PyTorch 설치 (CUDA 12.9 기준) 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
# 기타 라이브러리 설치 
pip install -r requirements.lock.txt


2. 실행
# OR-TOOLS Solver 실행
# (가상환경 활성화된 상태에서)
conda activate v7_env
# python -m [모듈이름] [설정파일] [옵션]
python3 -m or_tools_solver.main configs/config_6.json --max_sleep_current 0.001
python -m or_tools_solver.main config.json --max_sleep_current 0.01