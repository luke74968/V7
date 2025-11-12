# transformer_solver/definitions.py

from dataclasses import dataclass, field
from typing import List, Dict, Any



# --- 노드 "타입" (임베딩 및 구분용) ---
#
NODE_TYPE_PADDING = 0      # 0. (N_comp ~ N_max-1) 실제 문제에 포함되지 않는 슬롯
NODE_TYPE_BATTERY = 1      # 1. (Active)
NODE_TYPE_LOAD = 2         # 2. (Active)
NODE_TYPE_IC = 3           # 3. (Template 또는 Active) IC의 기본 타입
NODE_TYPE_EMPTY = 4        # 4. (Spawnable) 스폰될 수 있는 빈 슬롯

# --- 노드 피처 텐서 인덱스 (Lazy Spawn 상태 피처 추가) ---
FEATURE_INDEX = {
    "node_type": (0, 5),        # One-hot (5개): Padding, Battery, Load, IC, Empty
    
    # --- 피처 ---
    "cost": 5,
    "vin_min": 6,
    "vin_max": 7,
    "vout_min": 8,
    "vout_max": 9,
    "i_limit": 10,              # Pruning된 열-전기 복합 한계
    "current_active": 11,
    "current_sleep": 12,
    "current_out": 13,              # (동적) IC의 현재 총 출력 전류
    "ic_type_idx": 14,              # (정적) 0: N/A, 1: LDO, 2: Buck
    "op_current": 15,               # (정적) LDO 동작 전류
    "theta_ja": 16,                 # (정적) 열저항 (인덱스 -3)
    "t_junction_max": 17,           # (정적) 최대 허용 정션 온도 (인덱스 -3)
    "junction_temp": 18,            # (동적) 현재 정션 온도 (인덱스 -3)
    "quiescent_current": 19,        # (정적) 대기 전류 (인덱스 -3)
    "shutdown_current": 20,         # (정적) 차단 전류 (인덱스 -3)
    "independent_rail_type": 21,    # (정적) 0: 없음, 1: supplier, 2: path (인덱스 -3)
    "node_id": 22,                  # (정적) 노드 고유 ID (인덱스 -3)
    "always_on_in_sleep": 23,       # (정적) Always-On 요구 플래그 (인덱스 -3)
    
    # --- Lazy Spawn 상태 피처 ---
    "is_active": 24,       # (동적) 1.0 = 배터리, 로드, 스폰된(활성) IC (인덱스 -3)
    "is_template": 25,     # (정적) 1.0 = IC 템플릿 뱅크 (인덱스 -3)
    "can_spawn_into": 26,  # (정적) 1.0 = EMPTY 슬롯 (인덱스 -3)
    # ---------------------------------
}

FEATURE_DIM = 27
SCALAR_PROMPT_FEATURE_DIM = 4 # V6와 동일

@dataclass
class PocatConfig:
    """ 
    config.json의 내용을 템플릿 기준으로 담는 데이터 클래스 
    """
    battery: Dict[str, Any]
    available_ics: List[Dict[str, Any]] # 'IC 템플릿' 목록
    loads: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    
    node_names: List[str] = field(default_factory=list)
    node_types: List[int] = field(default_factory=list)

    def __post_init__(self):
        self.rebuild_node_lists()

    def rebuild_node_lists(self):
        """
        IC 템플릿 목록이 변경되었을 때 node_names와 node_types 리스트를 다시 생성합니다.
        """
        self.node_names.clear()
        self.node_types.clear()
        
        self.node_names.append(self.battery['name'])
        self.node_types.append(NODE_TYPE_BATTERY)

        # 2. Loads (IC보다 먼저 추가)
        for load in self.loads:
            self.node_names.append(load['name'])
            self.node_types.append(NODE_TYPE_LOAD)

        # 3. IC Templates (Load 뒤에 추가)
        for ic in self.available_ics:
            self.node_names.append(ic['name'])
            self.node_types.append(NODE_TYPE_IC) # 💡 'IC 템플릿'은 'IC' 타입으로 분류
