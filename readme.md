# 서버관련 셋업
# 가상환경 생성 (v6_env 라는 이름으로, Python 3.10 버전을 사용)
conda create -n v7_env python=3.12 -y
conda activate v7_env
# (가상환경이 (v7_env) 로 바뀐 것을 확인)
# PyTorch 설치 (CUDA 12.9 기준) 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
# 기타 라이브러리 설치 
pip install -r requirements.lock.txt

