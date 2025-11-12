# transformer_solver/expert_dataset.py

import json
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict
from tqdm import tqdm
from typing import Tuple, List, Dict, Any

from .solver_env import PocatEnv
from .env_generator import PocatGenerator

def expert_collate_fn(batch: List[Tuple[TensorDict, torch.Tensor]]) -> Tuple[TensorDict, torch.Tensor]:
    """
    TensorDict와 Tensor로 구성된 배치 리스트를
    하나의 스택(Stacked) 텐서로 묶는 커스텀 collate 함수입니다.
    
    Args:
        batch: [(TensorDict_1, Tensor_1), (TensorDict_2, Tensor_2), ...]
        
    Returns:
        (Stacked_TensorDict, Stacked_Tensor)
    """
    
    # 1. 튜플 리스트를 두 개의 리스트로 분리
    td_list = [item[0] for item in batch]
    reward_list = [item[1] for item in batch]
    
    # 2. TensorDict 리스트를 하나의 TensorDict로 스택
    # (B=1, N_max, D) -> (B=batch_size, N_max, D)
    batched_tds = torch.stack(td_list, dim=0)
    
    # 3. 보상 리스트를 (B, 1) 텐서로 스택
    batched_rewards = torch.stack(reward_list, dim=0)
    
    return batched_tds, batched_rewards


class ExpertReplayDataset(Dataset):
    """
    "정답지" JSON 파일을 로드하고, V7 환경에서 리플레이(Replay)하여
    (State, Final_Reward) 페어(Pair)를 생성하는 지도학습용 데이터셋입니다.
    
    (Critic 사전훈련용)
    """
    def __init__(self, 
                 expert_data_path: str, 
                 env: PocatEnv, 
                 device: str = "cpu",
                 N_max: int = 500):
        """
        Args:
            expert_data_path (str): "expert_data.json" 파일 경로.
            env (PocatEnv): 리플레이를 실행할 V7 환경 인스턴스.
            device (str): 텐서 디바이스.
            N_max (int): 환경의 N_MAX 값.
        """
        self.env = env
        self.device = device
        self.N_max = N_max
        self.replay_buffer: List[Tuple[TensorDict, torch.Tensor]] = []

        print(f"\n🧠 '정답지' 리플레이 데이터셋 생성 중 (Critic Pre-train)...")
        print(f"   - 정답지 파일 로드: {expert_data_path}")
        
        try:
            with open(expert_data_path, 'r', encoding='utf-8') as f:
                expert_traces = json.load(f)
            if not isinstance(expert_traces, list):
                expert_traces = []
        except Exception as e:
            print(f"❌ '정답지' 파일 로드 실패: {e}")
            expert_traces = []

        # (config.json 파일별로 V7 Generator를 캐싱)
        generator_cache: Dict[str, PocatGenerator] = {}

        pbar = tqdm(expert_traces, desc="   - OR-Tools 경로 리플레이 중")
        for trace in pbar:
            try:
                config_file = trace["config_file"]
                target_reward = trace["target_reward"]
                # V7 액션 시퀀스 (Parameterized Action 딕셔너리 리스트)
                action_sequences: List[List[Dict[str, Any]]] = trace["action_sequences"]
                
                # 1. 정답지와 동일한 config로 V7 Generator 준비
                if config_file not in generator_cache:
                    generator_cache[config_file] = PocatGenerator(
                        config_file_path=config_file,
                        N_max=self.N_max
                    )
                generator = generator_cache[config_file]
                
                # (B=1, 1) 크기의 정답 보상 텐서
                target_reward_tensor = torch.tensor([[target_reward]], dtype=torch.float32, device=self.device)

                # 2. 모든 경로(Load)를 순회
                for path_actions in action_sequences:
                    # 3. 환경 리셋 (B=1)
                    td_initial = generator(batch_size=1).to(self.device)
                    td = self.env._reset(td_initial) # (N_MAX 크기의 V7 상태)
                    
                    # 4. '정답지'의 V7 액션을 한 스텝씩 리플레이
                    for action_dict in path_actions:
                        
                        # (A) 리플레이: 현재 상태(td)와 정답 보상(target_reward)을 버퍼에 저장
                        # .clone()으로 텐서의 현재 스냅샷을 저장
                        self.replay_buffer.append((
                            td.clone().squeeze(0), # (1,N,D) -> (N,D)
                            target_reward_tensor.clone().squeeze(0) # (1,1) -> (1,)
                        ))
                        
                        # (B) 다음 스텝으로 이동 (V7 액션 딕셔너리를 텐서로 변환)
                        action_tensor_dict = {
                            "action_type": torch.tensor([[action_dict["action_type"]]], device=self.device),
                            "connect_target": torch.tensor([[action_dict["connect_target"]]], device=self.device),
                            "spawn_template": torch.tensor([[action_dict["spawn_template"]]], device=self.device),
                        }
                        
                        td.set("action", action_tensor_dict)
                        td = self.env.step(td)["next"]
                        
                        if td["done"].item():
                            break # (경로 완성 또는 실패)
                            
            except Exception as e:
                print(f"❌ 리플레이 중 오류 발생 (Config: {trace.get('config_file', 'N/A')}): {e}")
                
        if not self.replay_buffer:
            print("⚠️ 경고: '정답지' 리플레이 결과, 유효한 (상태, 보상) 데이터가 0개입니다.")
        else:
            print(f"✅ '정답지' 리플레이 완료. 총 {len(self.replay_buffer)}개의 (상태, 보상) 페어 생성.")

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def __getitem__(self, idx: int) -> Tuple[TensorDict, torch.Tensor]:
        """ 버퍼에서 (State, Reward) 페어를 반환합니다. """
        # (TensorDict[N_max, D], Tensor[1,])
        return self.replay_buffer[idx]