# V7-Pocat-Solver Readme

## 1. 초기 셋업

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


## 2. OR-Tools Solver 실행

# OR-TOOLS Solver 실행

# (가상환경 활성화된 상태에서)
conda activate v7_env

# python -m [모듈이름] [설정파일] [옵션]
python3 -m or_tools_solver.main configs/config_6.json --max_sleep_current 0.001


## 3. Transformer Solver (V7) 실행
# 사용할 GPU를 2번과 3번으로 지정
export CUDA_VISIBLE_DEVICES=2,3


### 훈련 (Training)

# 예시 1: config_6 문제로 훈련 (진행률 표시)
python3 -m transformer_solver.run --config_file configs/config_6.json --config_yaml configs/config.yaml --batch_size 256 --log_mode progress

# 예시 2: config_4 문제로 훈련 (상세 로그)
# V7은 N_MAX 아키텍처이므로 동일한 모델로 다른 크기의 문제를 훈련할 수 있습니다
python3 -m transformer_solver.run --config_file configs/config_4.json --config_yaml configs/config.yaml --batch_size 128 --log_mode progress

# 예시 3: Critic 사전훈련(Pre-training) 후 메인 훈련 시작
# (expert_data.json 파일이 준비되었다고 가정)
python3 -m transformer_solver.run --config_file configs/config_6.json --config_yaml configs/config.yaml --batch_size 256 --pretrain_critic expert_data/expert_data.json

### 추론 (Test)

# 훈련된 모델(.pth)을 사용하여 config_4 문제 풀기
python3 -m transformer_solver.run --test_only --config_file configs/config_4.json --config_yaml configs/config.yaml --log_mode detail --load_path "transformer_solver/result/YOUR_RUN_TIME/best_cost.pth"

# 동일한 모델로 N=218 config_6 문제 풀기
python3 -m transformer_solver.run --test_only --config_file configs/config_6.json --config_yaml configs/config.yaml --log_mode detail --load_path "transformer_solver/result/YOUR_RUN_TIME/best_cost.pth"

### 디버그 (Debug)

# config_6 문제로 대화형 디버거 실행
# (config.yaml의 N_MAX=500과 동일한 값을 --n_max로 전달)
python3 -m transformer_solver.debug_env configs/config_6.json --n_max 500