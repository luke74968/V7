# transformer_solver/solver_env.py

import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, UnboundedDiscrete, Composite
from typing import Optional, Dict, Union, Tuple, List

# --- 현재 패키지(transformer_solver) 모듈 임포트 ---
from .definitions import (
    FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM,
    NODE_TYPE_PADDING, NODE_TYPE_BATTERY, NODE_TYPE_LOAD, 
    NODE_TYPE_IC, NODE_TYPE_EMPTY
)
from .env_generator import PocatGenerator

# --- 환경 상수 ---
BATTERY_NODE_IDX = 0 # 배터리 노드는 항상 0번 인덱스
REWARD_WEIGHT_ACTION = 0.0  # (A2C) 액션(IC 스폰) 즉시 비용에 대한 가중치
REWARD_WEIGHT_PATH = 1.0    # (A2C) 경로(Load->BATT) 완성 시 누적 비용 가중치
STEP_PENALTY = 0.0          # (A2C) 스텝당 페널티
FAILURE_PENALTY = -500.0    # (A2C) 실패(막다른 길) 페널티
PENALTY_WEIGHT_SLEEP = 1000.0 # (A2C) 암전류 초과 페널티 가중치


class PocatEnv(EnvBase):
    """
    Pocat 문제를 풀기 위한 강화학습 환경(Environment)입니다.
    
    TensorDict를 상태(State)로 사용하며, Parameterized Action(Connect/Spawn)을
    처리하여 상태를 전이시키고 보상(Reward)을 계산합니다.
    """
    
    name = "pocat_env"

    def __init__(self, generator_params: dict, device: str = "cpu", N_max: int = 500, **kwargs):
        """
        PocatEnv를 초기화합니다.
        
        Args:
            generator_params (dict): PocatGenerator에 전달될 파라미터
            device (str): 텐서 연산을 수행할 디바이스
            N_max (int): 모델이 처리할 고정된 최대 노드 크기
        """
        super().__init__(device=device)
        
        # 1. N_max 값을 저장합니다.
        self.N_max = N_max
        
        # 2. 제너레이터 초기화 (N_max 전달)
        self.generator = PocatGenerator(**generator_params, N_max=N_max)
        
        # 3. 마스킹 및 계산에 사용할 버퍼 등록
        self.register_buffer("arange_nodes", None, persistent=False)
        self.register_buffer("node_type_tensor", None, persistent=False)
        self.register_buffer("load_idx_tensor", None, persistent=False)
        self.register_buffer("rail_types", None, persistent=False)

        # 4. Observation, Action 스펙 정의
        self._make_spec()
        
        # 5. 제약조건(시퀀싱, 독립) 정보 로드
        self._load_constraint_info()

    def _make_spec(self):
        """환경의 Observation, Action, Reward 스펙을 정의합니다."""
        
        num_nodes = self.N_max
        
        # 1. Observation 스펙 정의
        self.observation_spec = Composite({
            # --- 정적 텐서 (Generator 제공) ---
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM)),
            "scalar_prompt_features": Unbounded(shape=(SCALAR_PROMPT_FEATURE_DIM,)),
            "matrix_prompt_features": Unbounded(shape=(num_nodes, num_nodes)),
            "connectivity_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "attention_mask": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            
            # --- 동적 텐서 (Env 관리) ---
            "adj_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "adj_matrix_T": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "unconnected_loads_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "trajectory_head": UnboundedDiscrete(shape=(1,)),
            "step_count": UnboundedDiscrete(shape=(1,)),
            "current_cost": Unbounded(shape=(1,)),
            "staging_cost": Unbounded(shape=(1,)), # 현재 경로의 누적 비용
            "is_used_ic_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "current_target_load": UnboundedDiscrete(shape=(1,)),
            "is_exclusive_mask": Unbounded(shape=(num_nodes,), dtype=torch.long),
            "next_empty_slot_idx": UnboundedDiscrete(shape=(1,)), # 다음 스폰 위치
        })
        
        # 2. Action 스펙 정의 (Parameterized Action)
        self.action_spec = Composite({
            # (0: Connect, 1: Spawn)
            "action_type": UnboundedDiscrete(shape=(1,)),
            # (0 ~ N_max-1): Connect 대상
            "connect_target": UnboundedDiscrete(shape=(1,)),
            # (0 ~ N_max-1): Spawn할 템플릿
            "spawn_template": UnboundedDiscrete(shape=(1,)),
        })
        
        # 3. Reward 스펙 정의
        self.reward_spec = Unbounded(shape=(1,))

    def _load_constraint_info(self):
        """
        config 파일에서 제약조건 정보를 로드하고
        마스킹에 사용하기 쉽도록 텐서로 변환하여 저장합니다.
        """
        # (B,L,IC) 순서가 텐서 순서와 일치
        self.node_name_to_idx = {name: i for i, name in enumerate(self.generator.config.node_names)}
        
        # 1. Independent Rail (독립 레일) 정보
        rail_type_map = {"exclusive_supplier": 1, "exclusive_path": 2}
        rail_types_list = []
        
        # (Load 노드는 1번 인덱스부터 시작)
        load_start_idx = self.generator.num_battery
        for i, load_cfg in enumerate(self.generator.config.loads):
            load_idx = load_start_idx + i
            rail_type = rail_type_map.get(load_cfg.get("independent_rail_type"), 0)
            rail_types_list.append((load_idx, rail_type))
        
        # (N_max,) 크기의 텐서로 변환
        self.rail_types_tensor = torch.zeros(self.N_max, dtype=torch.long, device=self.device)
        if rail_types_list:
            indices = torch.tensor([idx for idx, _ in rail_types_list], dtype=torch.long, device=self.device)
            values = torch.tensor([val for _, val in rail_types_list], dtype=torch.long, device=self.device)
            self.rail_types_tensor.scatter_(0, indices, values)

        # 2. Power Sequence (전원 시퀀싱) 정보
        self.power_sequences = []
        for seq in self.generator.config.constraints.get("power_sequences", []):
            f_flag = seq.get("f", 1) # (1: 동일 부모 금지)
            j_idx = self.node_name_to_idx.get(seq['j'])
            k_idx = self.node_name_to_idx.get(seq['k'])
            if j_idx is not None and k_idx is not None:
                self.power_sequences.append((j_idx, k_idx, f_flag))

    def _ensure_buffers(self, td: TensorDict):
        """
        Observation 텐서가 변경될 때마다(주로 _reset 시)
        마스킹 계산에 필요한 헬퍼 텐서들을 미리 계산합니다.
        """
        num_nodes = td["nodes"].shape[1] # (N_max)

        if self.arange_nodes is None or self.arange_nodes.numel() != num_nodes:
            self.arange_nodes = torch.arange(num_nodes, device=self.device)
        
        if self.node_type_tensor is None:
            # (N_max,)
            node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
            self.node_type_tensor = node_types
            
            # (N_loads,)
            self.load_idx_tensor = torch.where(node_types == NODE_TYPE_LOAD)[0]

    def select_start_nodes(self, td: TensorDict) -> Tuple[int, torch.Tensor]:
        """ POMO (Multi-Start)를 위해 시작 가능한 모든 Load 노드의 인덱스를 반환합니다. """
        self._ensure_buffers(td) # load_idx_tensor 최신화
        return len(self.load_idx_tensor), self.load_idx_tensor

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """ 환경을 초기 상태(State)로 리셋합니다. """
        
        batch_size = kwargs.get("batch_size", self.batch_size)
        if td is None:
            if isinstance(batch_size, tuple): batch_size = batch_size[0]
            td_initial = self.generator(batch_size=batch_size).to(self.device)
        else:
            td_initial = td
            batch_size = td_initial.batch_size[0]

        num_nodes = self.N_max

        # --- 1. 동적 상태 텐서 초기화 ---
        
        # adj_matrix: (B, N_max, N_max) - 실제 연결된 엣지 (모두 0)
        adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)
        adj_matrix_T = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)

        # is_active_mask: (B, N_max) - 현재 활성화된 노드 (정적 피처에서 복사)
        is_active_mask = td_initial["nodes"][..., FEATURE_INDEX["is_active"]].bool()
        # (정적 피처에서 동적 마스크로 복사)
        is_template_mask = td_initial["nodes"][..., FEATURE_INDEX["is_template"]].bool()
        can_spawn_into_mask = td_initial["nodes"][..., FEATURE_INDEX["can_spawn_into"]].bool()

        # unconnected_loads_mask: (B, N_max) - 아직 연결 안 된 로드
        node_types = td_initial["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        unconnected_loads_mask = (node_types == NODE_TYPE_LOAD).unsqueeze(0).expand(batch_size, -1)
        
        # next_empty_slot_idx: (B, 1) - 다음 스폰 위치 (BATT+LOADS+TEMPLATES 개수)
        next_empty_slot_idx = torch.full((batch_size, 1), self.generator.num_components, dtype=torch.long, device=self.device)

        # 2. TensorDict 생성
        reset_td = TensorDict({
            # 정적 텐서 (Generator로부터 복사)
            "nodes": td_initial["nodes"].clone(),
            "scalar_prompt_features": td_initial["scalar_prompt_features"],
            "matrix_prompt_features": td_initial["matrix_prompt_features"],
            "connectivity_matrix": td_initial["connectivity_matrix"],
            "attention_mask": td_initial["attention_mask"],
            
            # 동적 텐서 (초기화)
            "adj_matrix": adj_matrix,
            "adj_matrix_T": adj_matrix_T,
            "unconnected_loads_mask": unconnected_loads_mask,
            "is_active_mask": is_active_mask,
            "is_template_mask": is_template_mask,
            "can_spawn_into_mask": can_spawn_into_mask,
            "next_empty_slot_idx": next_empty_slot_idx,
            "trajectory_head": torch.full((batch_size, 1), BATTERY_NODE_IDX, dtype=torch.long, device=self.device),
            "step_count": torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
            "current_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device),
            "staging_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device),
            "is_used_ic_mask": torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "current_target_load": torch.full((batch_size, 1), -1, dtype=torch.long, device=self.device),
            "is_exclusive_mask": torch.zeros(batch_size, num_nodes, dtype=torch.long, device=self.device),
            "done": torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
        }, batch_size=[batch_size], device=self.device)
       
        self._ensure_buffers(reset_td)
        return reset_td

    def step(self, tensordict: TensorDict) -> TensorDict:
        """ _step을 호출 (torchrl EnvBase 호환용) """
        return self._step(tensordict)

    def _step(self, td: TensorDict) -> TensorDict:
        """
        모델이 결정한 Parameterized Action을 실행하여
        환경의 상태(State)를 다음 스텝으로 전이시킵니다.
        
        - (액션 타입 0: Connect) -> 기존 활성 노드에 연결
        - (액션 타입 1: Spawn)   -> 템플릿을 Empty 슬롯에 복사(Spawn) 후 연결
        """
        batch_size, num_nodes = td["nodes"].shape[0], self.N_max
        action_dict = td["action"]
        
        # (B,)
        action_type = action_dict["action_type"].squeeze(-1)
        connect_target = action_dict["connect_target"].squeeze(-1)
        spawn_template = action_dict["spawn_template"].squeeze(-1)
        
        current_head = td["trajectory_head"].clone().squeeze(-1) # (B,)

        # --- 0. 이미 'done'인 배치는 무시 ---
        is_already_done = td["done"].squeeze(-1)
        if is_already_done.all():
            return TensorDict({
                "next": td, 
                "reward": torch.zeros(batch_size, 1, device=self.device), 
                "done": td["done"]}, batch_size=td.batch_size)

        # --- 1. 상태 텐서 복제 (수정 준비) ---
        next_obs = td.clone(recurse=False)
        # (In-place 수정이 발생하는 동적 텐서들은 모두 깊은 복사)
        next_obs["nodes"] = td["nodes"].clone() # (가장 중요)
        next_obs["adj_matrix"] = td["adj_matrix"].clone()
        next_obs["adj_matrix_T"] = td["adj_matrix_T"].clone()
        next_obs["connectivity_matrix"] = td["connectivity_matrix"].clone()
        next_obs["is_active_mask"] = td["is_active_mask"].clone()
        next_obs["is_template_mask"] = td["is_template_mask"].clone()
        next_obs["can_spawn_into_mask"] = td["can_spawn_into_mask"].clone()
        next_obs["current_target_load"] = td["current_target_load"].clone()
        next_obs["is_exclusive_mask"] = td["is_exclusive_mask"].clone()
        next_obs["staging_cost"] = td["staging_cost"].clone()
        next_obs["is_used_ic_mask"] = td["is_used_ic_mask"].clone()
        next_obs["adj_matrix_T"] = td["adj_matrix_T"].clone()
        next_obs["trajectory_head"] = td["trajectory_head"].clone()
        next_obs["unconnected_loads_mask"] = td["unconnected_loads_mask"].clone()
        next_obs["current_cost"] = td["current_cost"].clone()
        next_obs["next_empty_slot_idx"] = td["next_empty_slot_idx"].clone()

        step_reward = torch.full((batch_size,), STEP_PENALTY, dtype=torch.float32, device=self.device)
        batch_indices = torch.arange(batch_size, device=self.device)

        # --- 2. 액션 타입 분기 ---
        
        # --- 2a. [Select New Load] ---
        # (현재 헤드가 배터리일 때)
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            # 'Select New Load' 액션은 'Connect' 액션으로 전달됨
            b_idx_batt = batch_indices[head_is_battery]
            selected_load = connect_target[head_is_battery]

            is_load_selection = (selected_load != BATTERY_NODE_IDX)
            if is_load_selection.any():
                load_rows = b_idx_batt[is_load_selection]
                load_node_idx = selected_load[is_load_selection]
                
                # Head를 선택된 Load로 이동
                next_obs["trajectory_head"][load_rows, 0] = load_node_idx
                # '연결 안 됨' 마스크에서 제거
                next_obs["unconnected_loads_mask"][load_rows, load_node_idx] = False
                # 현재 경로의 최종 타겟으로 설정
                next_obs["current_target_load"][load_rows, 0] = load_node_idx
                # 경로 비용 초기화
                next_obs["staging_cost"][load_rows] = 0.0
                
                # '독립 레일' 상태 전파 시작
                # (B_load,)
                rail_status = self.rail_types_tensor[load_node_idx]
                next_obs["is_exclusive_mask"][load_rows, load_node_idx] = rail_status

        # --- 2b. [Find Parent / Spawn] ---
        # (현재 헤드가 Load 또는 IC일 때)
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = batch_indices[head_is_node]
            child_node = current_head[head_is_node] # (B_node,)
            
            # (B_node,)
            is_connect = (action_type[head_is_node] == 0)
            is_spawn = ~is_connect
            
            # (B_node,) - 최종 부모가 될 노드의 인덱스
            parent_node = torch.zeros_like(child_node)
            
            # --- Connect 액션 처리 ---
            if is_connect.any():
                b_idx_connect = b_idx_node[is_connect]
                # 'Connect' 헤드에서 부모 인덱스 가져오기
                parent_connect = connect_target[b_idx_connect]
                parent_node[is_connect] = parent_connect
            
            # --- Spawn 액션 처리 ---
            if is_spawn.any():
                b_idx_spawn = b_idx_node[is_spawn]
                child_spawn = child_node[is_spawn]
                
                # 'Spawn' 헤드에서 템플릿 인덱스 가져오기
                template_idx = spawn_template[b_idx_spawn] # (B_spawn,)
                # 스폰될 빈 슬롯 인덱스 가져오기
                slot_idx = next_obs["next_empty_slot_idx"][b_idx_spawn].squeeze(-1) # (B_spawn,)
                
                # 1. Spawn: 템플릿 피처 -> 빈 슬롯으로 복사
                template_features = next_obs["nodes"][b_idx_spawn, template_idx]
                next_obs["nodes"][b_idx_spawn, slot_idx] = template_features.detach()

                # Spawn된 슬롯은 템플릿과 동일한 전압 호환성을 가져야 하므로
                # connectivity_matrix의 행/열을 템플릿에서 복사한다.
                connectivity_matrix = next_obs["connectivity_matrix"]
                connectivity_matrix[b_idx_spawn, :, slot_idx] = connectivity_matrix[b_idx_spawn, :, template_idx]
                connectivity_matrix[b_idx_spawn, slot_idx, :] = connectivity_matrix[b_idx_spawn, template_idx, :]

                # 2. 상태 변경: (Template -> Active)
                next_obs["nodes"][b_idx_spawn, slot_idx, FEATURE_INDEX["is_active"]] = 1.0
                next_obs["nodes"][b_idx_spawn, slot_idx, FEATURE_INDEX["is_template"]] = 0.0
                next_obs["nodes"][b_idx_spawn, slot_idx, FEATURE_INDEX["can_spawn_into"]] = 0.0
                
                # 3. 환경 동적 마스크 업데이트
                next_obs["is_active_mask"][b_idx_spawn, slot_idx] = True
                next_obs["is_template_mask"][b_idx_spawn, slot_idx] = False
                next_obs["can_spawn_into_mask"][b_idx_spawn, slot_idx] = False

                # 4. 다음 빈 슬롯 인덱스 +1
                next_obs["next_empty_slot_idx"][b_idx_spawn] += 1
                
                # 5. 스폰 비용(cost) 즉시 반영
                template_cost = next_obs["nodes"][b_idx_spawn, template_idx, FEATURE_INDEX["cost"]]
                staging_cost_increase = template_cost.unsqueeze(-1) # (B_spawn, 1)
                
                # staging_cost 및 current_cost에 스폰 비용 추가
                next_obs["staging_cost"][b_idx_spawn] += staging_cost_increase
                next_obs["current_cost"][b_idx_spawn] += staging_cost_increase
                
                # R_action 보상 (스폰 즉시)
                step_reward[b_idx_spawn] += REWARD_WEIGHT_ACTION * (-staging_cost_increase.squeeze(-1))
                
                # 'is_used_ic_mask'에 템플릿 인덱스 대신 *스폰된 슬롯 인덱스*를 기록
                next_obs["is_used_ic_mask"][b_idx_spawn, slot_idx] = True
                
                # 6. 최종 부모를 '스폰된 슬롯'으로 설정
                parent_node[is_spawn] = slot_idx

            # --- 3. 공통 연결 로직 (Connect/Spawn 공통) ---
            
            # 3a. 엣지 추가: (parent_node) -> (child_node)
            next_obs["adj_matrix"][b_idx_node, parent_node, child_node] = True
            # (T에도 엣지 추가: (child_node) -> (parent_node))
            next_obs["adj_matrix_T"][b_idx_node, child_node, parent_node] = True

            # 3b. '독립 레일' 상태 전파 (자식 -> 부모)
            child_status = next_obs["is_exclusive_mask"][b_idx_node, child_node] # (B_node,)
            if (child_status > 0).any():
                parent_status = next_obs["is_exclusive_mask"][b_idx_node, parent_node]
                
                # 'Path'(2)는 IC를 타고 계속 전파됨
                status_to_propagate = torch.where(
                    child_status == 2, 
                    child_status, 
                    torch.tensor(0, device=self.device, dtype=torch.long)
                )
                
                # 'Supplier'(1)는 Load에서 시작할 때만 전파됨
                is_child_load = (self.node_type_tensor[child_node] == NODE_TYPE_LOAD)
                status_from_supplier = torch.where(
                    (child_status == 1) & is_child_load,
                    child_status,
                    torch.tensor(0, device=self.device, dtype=torch.long)
                )
                
                status_from_child = torch.max(status_to_propagate, status_from_supplier)
                new_parent_status = torch.max(parent_status, status_from_child)
                next_obs["is_exclusive_mask"][b_idx_node, parent_node] = new_parent_status

            # 3c. 다음 Head 설정
            parent_is_battery = (parent_node == BATTERY_NODE_IDX)
            
            # 헤드(parent_node)가 이미 부모를 가졌는지 확인
            # adj_matrix_T[b, node, :]가 1이라도 있으면, node는 이미 부모가 있음
            parent_already_has_parent = next_obs["adj_matrix_T"][b_idx_node, parent_node].any(dim=-1)
            
            # 배터리에 도달하거나, 이미 연결된 노드에 도달하면 경로 완성
            path_is_finished = parent_is_battery | parent_already_has_parent

            next_obs["trajectory_head"][b_idx_node, 0] = torch.where(
                path_is_finished,  # 💡 조건 변경
                BATTERY_NODE_IDX,  # 경로가 끝났으면 배터리로 복귀
                parent_node        # 아니면 경로 추적 계속
            )
            
            # 3d. 경로 완성 (R_path 보상)
            if parent_is_battery.any():
                finished_rows = b_idx_node[parent_is_battery]
                
                # 경로 완성 시, 누적된 staging_cost를 R_path 보상으로 추가
                sub_trajectory_total_cost = next_obs["staging_cost"][finished_rows]
                step_reward[finished_rows] += REWARD_WEIGHT_PATH * (-sub_trajectory_total_cost.squeeze(-1))
                
                # staging_cost 리셋
                next_obs["staging_cost"][finished_rows] = 0.0
                next_obs["current_target_load"][finished_rows, 0] = -1

        # --- 4. 전력/발열 재계산 (연산 비용 높음) ---
        # (모든 배치가 최소 1스텝 이상 진행했을 때만 계산)
        if td["step_count"].min() > 0 or head_is_node.any():
            final_i_out, power_loss, new_temp = self._calculate_tree_loads(
                next_obs["nodes"], 
                next_obs["adj_matrix"],
                next_obs["adj_matrix_T"] # 💡 adj_matrix_T 전달
            )
            next_obs["nodes"][..., FEATURE_INDEX["current_out"]] = final_i_out
            next_obs["nodes"][..., FEATURE_INDEX["junction_temp"]] = new_temp
        
        next_obs.set("step_count", td["step_count"] + 1)

        # --- 5. 종료 조건 확인 ---
        # (get_action_mask가 3종 마스크를 모두 반환한다고 가정)
        next_masks = self.get_action_mask(next_obs)
        # Connect/Spawn 둘 다 불가능한 경우
        is_stuck = ~(next_masks["mask_type"].any(dim=-1))
        
        all_loads_connected = (next_obs["unconnected_loads_mask"].sum(dim=1) == 0)
        trajectory_finished = (next_obs["trajectory_head"].squeeze(-1) == BATTERY_NODE_IDX)
        
        done_successfully = all_loads_connected & trajectory_finished
        max_steps = 2 * self.N_max # 최대 스텝 제한
        timed_out = (next_obs["step_count"] > max_steps).squeeze(-1)
        
        is_done = done_successfully | timed_out | is_stuck
        next_obs["done"] = is_done.unsqueeze(-1)

        # --- 6. 최종 보상 계산 ---
        final_reward = self.get_reward(
            next_obs,
            step_reward, # (STEP_PENALTY + R_action + R_path)
            done_successfully,
            timed_out,
            is_stuck
        )
        
        # 이미 'done'이었던 샘플은 보상 0, 상태 롤백
        if is_already_done.any():
            final_reward[is_already_done] = 0.0
            next_obs[is_already_done] = td[is_already_done]

        return TensorDict({
            "next": next_obs,
            "reward": final_reward.unsqueeze(-1),
            "done": next_obs["done"],
        }, batch_size=batch_size)
        
    def get_reward(self,
                   td: TensorDict,
                   step_reward: torch.Tensor,
                   done_successfully: torch.Tensor,
                   timed_out: torch.Tensor,
                   is_stuck: torch.Tensor) -> torch.Tensor:
        """
        최종 스텝 보상을 계산합니다.
        (기본 스텝 보상 + 성공 시 암전류 페널티 or 실패 시 페널티)
        """
        reward = step_reward.clone()

        # 1. 성공한 경우: 암전류 제약 검사
        if done_successfully.any():
            td_success = td[done_successfully]
            
            total_sleep_current = self._calculate_total_sleep_current(td_success)
            
            # (스칼라 프롬프트 1번 인덱스 = max_sleep_current)
            max_sleep_current = td_success["scalar_prompt_features"][:, 1]
            
            # Hinge Loss: 초과한 만큼만 페널티 적용
            violation_amount = total_sleep_current - max_sleep_current
            hinge_violation = torch.relu(violation_amount)
            
            sleep_penalty = PENALTY_WEIGHT_SLEEP * hinge_violation
            
            # reward에 페널티 차감
            reward[done_successfully] -= sleep_penalty

        # 2. 실패한 경우: 고정 페널티
        failed = (timed_out | is_stuck) & ~done_successfully
        if failed.any():
            reward[failed] = FAILURE_PENALTY
            
        return reward

    # ---
    # 섹션 5: 액션 마스킹 (연산 집약적)
    # ---
    
    def get_action_mask(self, td: TensorDict, debug: bool = False) -> Dict[str, torch.Tensor]:
        """
        현재 상태(td)에서 가능한 모든 액션을 계산하여
        3종류의 마스크 딕셔너리를 반환합니다.
        
        Returns:
            {
                "mask_type": (B, 2) - [Can_Connect, Can_Spawn]
                "mask_connect": (B, N_max) - 연결 가능한 부모(Active)
                "mask_spawn": (B, N_max) - 스폰 가능한 템플릿(Template)
            }
        """
        self._ensure_buffers(td) # 버퍼 최신화

        batch_size, num_nodes = td.batch_size[0], self.N_max
        current_head = td["trajectory_head"].clone().squeeze(-1) # (B,)

        # --- 1. 기본 상태 마스크 (저비용) ---
        is_active = td["is_active_mask"] # (B, N_max) - 현재 활성 노드
        is_template = td["is_template_mask"] # (B, N_max) - 템플릿 노드
        
        # --- 2. [Select New Load] 모드 마스킹 ---
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if debug:
            reasons = {}

        # (B, 2)
        mask_type = torch.zeros(batch_size, 2, dtype=torch.bool, device=self.device)
        # (B, N_max)
        mask_connect = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        # (B, N_max)
        mask_spawn = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)

        if head_is_battery.any():
            b_idx_batt = torch.where(head_is_battery)[0]
            
            # 배터리에서는 'Connect'만 가능
            mask_type[b_idx_batt, 0] = True 
            
            # 'Connect' 대상은 'unconnected_loads_mask'
            mask_connect[b_idx_batt] = td["unconnected_loads_mask"][b_idx_batt]
            
            # 만약 모든 로드가 연결되었으면 (unconnected.sum() == 0),
            # 'BATTERY_NODE_IDX' (0번)에 연결하여 종료 신호
            all_connected = (td["unconnected_loads_mask"][b_idx_batt].sum(dim=-1) == 0)
            if all_connected.any():
                b_idx_finish = b_idx_batt[all_connected]
                mask_connect[b_idx_finish, BATTERY_NODE_IDX] = True
        
            if debug:
                reasons["mask_type"] = mask_type[b_idx_batt]
                reasons["mask_connect"] = mask_connect[b_idx_batt]

        # --- 3. [Find Parent / Spawn] 모드 마스킹 (고비용) ---
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = torch.where(head_is_node)[0]
            child_nodes = current_head[b_idx_node] # (B_node,)
            B_act = len(b_idx_node) # 실제 연산할 배치 크기
            
            # --- 3a. 저비용 마스크 (전압, 사이클, 독립, 시퀀싱) ---
            
            # (B_act, N_max)
            connectivity = td["connectivity_matrix"][b_idx_node]
            volt_ok = connectivity[torch.arange(B_act), :, child_nodes]
            
            # (B_act, N_max)
            path_mask = self._trace_path_batch(child_nodes, td["adj_matrix_T"][b_idx_node])
            cycle_ok = ~path_mask            


            # (B_act, N_max)
            exclusive_ok = self._get_exclusive_mask(
                td,           # 💡 'td' 전달
                b_idx_node,   # 💡 'b_idx_node' 전달
                child_nodes
            )
            
            # (B_act, N_max)
            power_seq_ok = self._get_power_sequence_mask(
                td["adj_matrix"][b_idx_node],
                child_nodes,
                td,           
                b_idx_node    
            )
            
            # (B_act, N_max) - 모든 저비용 제약을 통과한 후보
            base_valid_parents = volt_ok & cycle_ok & exclusive_ok & power_seq_ok
            
            # --- 3b. 고비용 마스크 (전류/발열 시뮬레이션) ---
            
            # (B_act, N_max) - 시뮬레이션으로 검증된 최종 유효 부모
            thermal_current_ok = self._get_thermal_current_mask(
                td,
                b_idx_node,
                child_nodes,
                base_valid_parents # 시뮬레이션 대상을 줄이기 위해 저비용 마스크 전달
            )
            
            final_valid_parents = base_valid_parents & thermal_current_ok
            
            # --- 3c. 최종 3종 마스크 생성 ---
            
            # 'Connect' 대상: 최종 유효 부모 + 'Active' 상태
            mask_connect[b_idx_node] = final_valid_parents & is_active[b_idx_node]
            
            # 'Spawn' 대상: 최종 유효 부모 + 'Template' 상태
            mask_spawn[b_idx_node] = final_valid_parents & is_template[b_idx_node]
            
            # 'Type' 마스크
            can_connect = mask_connect[b_idx_node].any(dim=-1) # (B_node,)
            
            # (스폰은 빈 슬롯이 남아있어야 가능)
            has_empty_slots = (td["next_empty_slot_idx"][b_idx_node] < self.N_max)
            can_spawn = mask_spawn[b_idx_node].any(dim=-1) & has_empty_slots.squeeze(-1)
            
            mask_type[b_idx_node, 0] = can_connect
            mask_type[b_idx_node, 1] = can_spawn
            
            if debug:
                # (디버그 정보는 b_idx_node[0] (0번 샘플) 기준으로만 수집)
                if 0 in b_idx_node:
                    reasons["volt_ok"] = volt_ok[0]
                    reasons["cycle_ok"] = cycle_ok[0]
                    reasons["exclusive_ok"] = exclusive_ok[0]
                    reasons["power_seq_ok"] = power_seq_ok[0]
                    reasons["base_valid_parents"] = base_valid_parents[0]
                    reasons["thermal_current_ok"] = thermal_current_ok[0]
                    reasons["final_valid_parents"] = final_valid_parents[0]

        return {
            "mask_type": mask_type,
            "mask_connect": mask_connect,
            "mask_spawn": mask_spawn,
        }

        if debug:
            return {
                "mask_type": mask_type,
                "mask_connect": mask_connect,
                "mask_spawn": mask_spawn,
                "reasons": reasons # 디버그 정보 반환
            }
        else:
            return {
                "mask_type": mask_type,
                "mask_connect": mask_connect,
                "mask_spawn": mask_spawn,
            }

    # ---
    # 섹션 6: 마스킹 헬퍼 함수 (V6 로직 벡터화/적응)
    # ---

    def _trace_path_batch(self, start_nodes: torch.Tensor, adj_matrix_T: torch.Tensor) -> torch.Tensor:
        """ (V6 계승) start_nodes의 모든 조상(ancestors)을 찾아 마스크로 반환 (사이클 방지용) """
        batch_size, num_nodes, _ = adj_matrix_T.shape
        path_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)

        if start_nodes.numel() > 0:
            path_mask.scatter_(1, start_nodes.unsqueeze(-1), True)
        
        adj_matrix_T_float = adj_matrix_T.float()

        for _ in range(num_nodes):
            # (B,N,N) @ (B,N,1) -> (B,N)
            parents_mask = (adj_matrix_T_float @ path_mask.float().unsqueeze(-1)).squeeze(-1).bool()
            if (parents_mask & ~path_mask).sum() == 0: break
            path_mask |= parents_mask
            
        return path_mask

    def _get_exclusive_mask(self, 
                            td: TensorDict,           # 💡 'td' 인자 추가
                            b_idx_node: torch.Tensor, # 💡 'b_idx_node' 인자 추가
                            child_nodes: torch.Tensor
                            ) -> torch.Tensor:
        """ (V6 계승) 독립 레일(Exclusive Rail) 제약조건 마스크 생성 """
        # (td와 b_idx_node에서 필요한 텐서를 가져옴)
        is_exclusive_mask_batch = td["is_exclusive_mask"][b_idx_node]
        adj_matrix_batch = td["adj_matrix"][b_idx_node]
        B_act, N_nodes = is_exclusive_mask_batch.shape
        
        # 1. Head(Child)의 상태adj_matrix_T
        head_status = is_exclusive_mask_batch[torch.arange(B_act), child_nodes] # (B_act,)
        head_is_load = (self.node_type_tensor[child_nodes] == NODE_TYPE_LOAD) # (B_act,)

        # 2. Parent(후보)의 상태
        parent_status = is_exclusive_mask_batch # (B_act, N_nodes)
        parent_is_exclusive = (parent_status > 0)
        
        # (B_act, N_nodes) - 부모가 이미 자식을 가졌는지?
        parent_has_any_child = adj_matrix_batch.any(dim=-1)
        
        # 3. 위반(Violation) 규칙 (True = 위반 = 금지)
        
        # 규칙 1: Head가 'Path'(2) -> 부모는 자식이 없어야 함
        # (B_act, 1) & (B_act, N_nodes) -> (B_act, N_nodes)
        violation_Rule1 = (head_status.unsqueeze(-1) == 2) & parent_has_any_child
        
        # 규칙 2: Head가 'Supplier Load'(1) -> 부모는 자식이 없어야 함
        violation_Rule2 = ((head_status == 1) & head_is_load).unsqueeze(-1) & parent_has_any_child
        
        # 규칙 3: Head가 'Normal'(0) 또는 'Supplier IC'(1) -> 부모는 Exclusive이면 안 됨
        violation_Rule3 = ((head_status == 0) | ((head_status == 1) & ~head_is_load)).unsqueeze(-1) & parent_is_exclusive
        
        violations = violation_Rule1 | violation_Rule2 | violation_Rule3
        
        # 4. 배터리는 항상 허용
        is_battery_mask = (self.arange_nodes.unsqueeze(0) == BATTERY_NODE_IDX)
        exclusive_ok = torch.logical_not(violations) | is_battery_mask
        
        return exclusive_ok

    def _get_power_sequence_mask(self,
                                 adj_matrix_batch: torch.Tensor,
                                 child_nodes: torch.Tensor,
                                 td: TensorDict,           
                                 b_idx_node: torch.Tensor  
                                 ) -> torch.Tensor:
        """ 전원 시퀀싱(Power Sequence) 제약조건 마스크 생성 """
        B_act, N_nodes, _ = adj_matrix_batch.shape
        adj_matrix_T_batch = td["adj_matrix_T"][b_idx_node]
        candidate_mask = torch.ones(B_act, N_nodes, dtype=torch.bool, device=self.device)

        for j_idx, k_idx, f_flag in self.power_sequences:
            # Case 1: 현재 child가 'k' (j의 부모를 찾는 중)
            is_k_mask = (child_nodes == k_idx)
            if is_k_mask.any():
                b_idx_check = torch.where(is_k_mask)[0] # (B_k,)
                
                # 'j'의 부모가 이미 존재하는가?
                parent_of_j_exists = adj_matrix_batch[b_idx_check, :, j_idx].any(dim=-1) # (B_k,)
                
                if parent_of_j_exists.any():
                    b_constr = b_idx_check[parent_of_j_exists] # (B_constr,)
                    
                    # 'j'의 부모 인덱스 (V6는 argmax, V7은 Full AdjT)
                    parent_of_j_idx = adj_matrix_batch[b_constr, :, j_idx].long().argmax(-1) # (B_constr,)
                    
                    # 'j'의 부모(parent_of_j)의 모든 조상(ancestors)을 찾음
                    anc_mask = self._trace_path_batch(parent_of_j_idx, adj_matrix_T_batch[b_constr])
                    anc_mask[:, BATTERY_NODE_IDX] = False # 배터리 제외
                    
                    # 'k'의 부모는 'j'의 조상이 될 수 없음
                    candidate_mask[b_constr] &= ~anc_mask
                    
                    # (f=1) 'k'의 부모는 'j'의 부모와 같을 수 없음
                    if f_flag == 1:
                        same_parent_mask = (self.arange_nodes == parent_of_j_idx.unsqueeze(1))
                        candidate_mask[b_constr] &= ~same_parent_mask

            # Case 2: (V6와 동일) 현재 child가 'j' (k의 부모를 찾는 중)
            is_j_mask = (child_nodes == j_idx)
            if is_j_mask.any():
                b_idx_check = torch.where(is_j_mask)[0]
                parent_of_k_exists = adj_matrix_batch[b_idx_check, :, k_idx].any(dim=-1)
                
                if parent_of_k_exists.any():
                    b_constr = b_idx_check[parent_of_k_exists]
                    parent_of_k_idx = adj_matrix_batch[b_constr, :, k_idx].long().argmax(-1)
                    
                    anc_mask = self._trace_path_batch(parent_of_k_idx, adj_matrix_T_batch[b_constr])
                    anc_mask[:, BATTERY_NODE_IDX] = False
                    
                    # 'j'의 부모는 'k'의 조상이 될 수 없음
                    candidate_mask[b_constr] &= ~anc_mask
                    
                    if f_flag == 1:
                        same_parent_mask = (self.arange_nodes == parent_of_k_idx.unsqueeze(1))
                        candidate_mask[b_constr] &= ~same_parent_mask
                        
        return candidate_mask

    def _get_thermal_current_mask(self,
                                  td: TensorDict,
                                  b_idx_node: torch.Tensor,
                                  child_nodes: torch.Tensor,
                                  base_valid_parents: torch.Tensor) -> torch.Tensor:
        """
        (V6 계승) 전류/발열 한계를 만족하는지 시뮬레이션하여 마스크 생성.
        (연산 비용이 가장 높은 함수)
        """
        
        # (B_act, N_max)
        thermal_current_ok = base_valid_parents.clone()
        
        # 시뮬레이션 청크 크기 (메모리/속도 트레이드오프)
        SIM_CHUNK_SIZE = 32 
        
        B_act, N_nodes = base_valid_parents.shape
        
        base_nodes = td["nodes"][b_idx_node]
        base_adj_matrix = td["adj_matrix"][b_idx_node]
        base_adj_matrix_T = td["adj_matrix_T"][b_idx_node]
        
        # 마진(Margin) 값 미리 로드
        margin_I = float(self.generator.config.constraints.get("current_margin", 0.0))
        margin_T = float(self.generator.config.constraints.get("thermal_margin_percent", 0.0))
        
        # (N_max,) - IC 타입 마스크
        ic_mask_1d = (self.node_type_tensor == NODE_TYPE_IC)
        
        # (N_max) 크기의 청크로 나누어 시뮬레이션
        for chunk_start in range(0, N_nodes, SIM_CHUNK_SIZE):
            chunk_end = min(chunk_start + SIM_CHUNK_SIZE, N_nodes)
            parent_indices_in_chunk = torch.arange(chunk_start, chunk_end, device=self.device)
            
            # (B_act, N_chunk) - 이번 청크에서 시뮬레이션할 (배치, 부모) 후보
            candidates_in_chunk_mask = base_valid_parents[:, chunk_start:chunk_end]
            
            # (N_sim,) - (B_act 기준 인덱스, 로컬 부모 인덱스)
            b_idx_sim_chunk, p_idx_sim_chunk_local = candidates_in_chunk_mask.nonzero(as_tuple=True)
            
            if b_idx_sim_chunk.numel() == 0:
                continue # 시뮬레이션할 후보 없음
            
            N_sim = b_idx_sim_chunk.numel()
            
            # 1. 시뮬레이션 데이터 준비 (N_sim,)
            sim_nodes = base_nodes[b_idx_sim_chunk]
            sim_adj_matrix = base_adj_matrix[b_idx_sim_chunk].clone()
            sim_adj_matrix_T = base_adj_matrix_T[b_idx_sim_chunk].clone()
            sim_child_nodes = child_nodes[b_idx_sim_chunk]
            
            # (N_sim,) - 실제 부모 노드 인덱스
            sim_parent_indices_global = parent_indices_in_chunk[p_idx_sim_chunk_local]
            
            # 2. (가상) 엣지 추가: (parent) -> (child)
            sim_rows = torch.arange(N_sim, device=self.device)
            sim_adj_matrix[sim_rows, sim_parent_indices_global, sim_child_nodes] = True
            sim_adj_matrix_T[sim_rows, sim_child_nodes, sim_parent_indices_global] = True

            # 3. 🚀 트리 전체 부하 시뮬레이션
            (final_i_out, power_loss, junction_temp) = self._calculate_tree_loads(
                sim_nodes, 
                sim_adj_matrix,
                sim_adj_matrix_T # 💡 T 매트릭스 전달
            )

            # 4. 시뮬레이션 결과 검증
            i_limit_raw = sim_nodes[..., FEATURE_INDEX["i_limit"]]
            t_max_raw   = sim_nodes[..., FEATURE_INDEX["t_junction_max"]]
            
            i_limit = i_limit_raw * (1.0 - margin_I)
            t_max   = t_max_raw   # (V6는 T_max에 마진 적용 안 함, V7 common은 적용함. V6 계승)
            
            # (N_sim, N_max)
            current_check_ok = (final_i_out <= i_limit + 1e-6)
            temp_check_ok = (junction_temp <= t_max + 1e-6)
            
            # (N_sim, N_max)
            all_checks_ok = current_check_ok & temp_check_ok
            
            # (N_sim, N_max) - IC가 아닌 노드는 항상 OK
            ic_mask_sim = ic_mask_1d.expand(N_sim, -1)
            
            # (N_sim,) - (모든 IC가 OK)
            is_valid_simulation = (all_checks_ok | ~ic_mask_sim).all(dim=-1)

            # 5. 실패한 시뮬레이션 결과를 (B_act, N_max) 마스크에 반영
            failed_sim_mask = ~is_valid_simulation
            if failed_sim_mask.any():
                b_idx_failed = b_idx_sim_chunk[failed_sim_mask]
                p_idx_failed_global = sim_parent_indices_global[failed_sim_mask]
                
                thermal_current_ok[b_idx_failed, p_idx_failed_global] = False
                
        return thermal_current_ok

    # ---
    # 섹션 7: 계산 헬퍼 함수 (V6 로직 벡터화/적응)
    # ---

    def _calculate_tree_loads(self, 
                              nodes_tensor: torch.Tensor, 
                              adj_matrix: torch.Tensor,
                              adj_matrix_T: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ (V6 계승) Adjacency Matrix를 기반으로 트리 전체의 전류/전력손실/온도를 계산합니다. """
        
        batch_size, num_nodes, _ = nodes_tensor.shape
        
        # 1. 초기 수요 = Load의 활성 전류
        current_demands = nodes_tensor[..., FEATURE_INDEX["current_active"]].clone()
        
        # Load가 아닌 노드는 초기 수요가 0
        load_mask_1d = (self.node_type_tensor == NODE_TYPE_LOAD) # (N_max,)
        current_demands[:, ~load_mask_1d] = 0.0
        
        adj_matrix_float = adj_matrix.float()
        adj_matrix_T_float = adj_matrix_T.float()

        # (IC 타입, LDO/Buck)
        ic_type = nodes_tensor[..., FEATURE_INDEX["ic_type_idx"]]
        ldo_mask_b = torch.isclose(ic_type, torch.tensor(1.0, device=ic_type.device))
        buck_mask_b = torch.isclose(ic_type, torch.tensor(2.0, device=ic_type.device))
        
        op_current = nodes_tensor[..., FEATURE_INDEX["op_current"]]
        vout = nodes_tensor[..., FEATURE_INDEX["vout_min"]] # (고정 Vout)
        vin = nodes_tensor[..., FEATURE_INDEX["vin_min"]]  # (고정 Vin)
        safe_vin = torch.where(vin > 0, vin, 1e-6)
        eff = 0.9 # (고정 효율)

        # 2. 전류 전파 (Bottom-up)
        # (i_in_total은 (B, N_max)로, 각 노드가 *소비*하는 총 전류)
        i_in_total = current_demands.clone() 

        for _ in range(num_nodes):
            # i_out (B, N_max) = 이 노드가 자식들에게 *공급*해야 하는 총 전류
            # i_out = (B,N,N) @ (B,N,1) -> (B,N)
            i_out = (adj_matrix_float @ i_in_total.unsqueeze(-1)).squeeze(-1)            

            # I_in_ldo/buck (B, N_max) = 이 노드가 *공급*하기 위해 *소비*해야 하는 전류
            i_in_ldo = i_out + op_current
            
            # Buck 입력 전류: I_in = P_out / (Eff * V_in) + I_op
            p_out_buck = vout * i_out
            i_in_buck = (p_out_buck / eff) / safe_vin + op_current
            
            # (B, N_max) - IC 노드들의 총 입력 수요
            new_ic_demands = torch.zeros_like(i_in_total)
            new_ic_demands[ldo_mask_b] = i_in_ldo[ldo_mask_b]
            new_ic_demands[buck_mask_b] = i_in_buck[buck_mask_b]

            new_i_in_total = current_demands + new_ic_demands
            if torch.allclose(i_in_total, new_i_in_total, atol=1e-8):
                break
            i_in_total = new_i_in_total

        i_out = (adj_matrix_float @ i_in_total.unsqueeze(-1)).squeeze(-1)
            
        # 3. 최종 손실 및 온도 계산
        power_loss = self._calculate_power_loss(
            nodes_tensor, i_out, ldo_mask_b, buck_mask_b
        )
        theta_ja = nodes_tensor[..., FEATURE_INDEX["theta_ja"]]
        ambient_temp = self.generator.config.constraints.get("ambient_temperature", 25.0)
        junction_temp = ambient_temp + power_loss * theta_ja
        
        return i_out, power_loss, junction_temp

    def _calculate_power_loss(self, 
                              ic_node_features: torch.Tensor, 
                              i_out: torch.Tensor,
                              ldo_mask: torch.Tensor,
                              buck_mask: torch.Tensor
                              ) -> torch.Tensor:
        """ (V6 계승) I_out을 기반으로 IC의 전력 손실(P_loss)을 계산합니다. """
        
        vin = ic_node_features[..., FEATURE_INDEX["vin_min"]]
        vout = ic_node_features[..., FEATURE_INDEX["vout_min"]]
        op_current = ic_node_features[..., FEATURE_INDEX["op_current"]]
        power_loss = torch.zeros_like(i_out)
        
        # LDO: P_loss = (V_in - V_out) * I_out + V_in * I_op
        if ldo_mask.any():
            power_loss[ldo_mask] = (vin[ldo_mask] - vout[ldo_mask]) * i_out[ldo_mask] + \
                                   vin[ldo_mask] * op_current[ldo_mask]
        
        # Buck: P_loss = P_out * (1/Eff - 1) + V_in * I_op
        if buck_mask.any():
            p_out_buck = vout[buck_mask] * i_out[buck_mask]
            eff = 0.9 # (고정 효율)
            conversion_loss = (p_out_buck / eff) - p_out_buck
            power_loss[buck_mask] = conversion_loss + vin[buck_mask] * op_current[buck_mask]
            
        return power_loss

    def _calculate_total_sleep_current(self, td: TensorDict) -> torch.Tensor:
        """ (V6 계승) 암전류(Sleep Current) 제약조건을 검사합니다. """
        
        batch_size, num_nodes, _ = td["nodes"].shape
        adj_matrix = td["adj_matrix"].float()
        adj_matrix_T = td["adj_matrix_T"].float() # 💡 (c, p) -> (p, c)

        # 1. "Always-On" 상태 전파 (Load -> Battery)
        always_on_loads = (td["nodes"][..., FEATURE_INDEX["always_on_in_sleep"]] == 1.0)
        always_on_nodes = always_on_loads.clone()
        always_on_nodes[:, BATTERY_NODE_IDX] = True
        
        for _ in range(num_nodes):
            parents_mask = (adj_matrix_T @ always_on_nodes.float().unsqueeze(-1)).squeeze(-1).bool()
            if (parents_mask & ~always_on_nodes).sum() == 0: break
            always_on_nodes |= parents_mask
        
        # 2. IC 자체 암전류 소모 (3-State)
        is_ao = always_on_nodes
        is_used = td["is_used_ic_mask"]
        parent_is_ao = (adj_matrix_T @ is_ao.float().unsqueeze(-1)).squeeze(-1).bool()

        op_current = td["nodes"][..., FEATURE_INDEX["op_current"]]
        quiescent_current = td["nodes"][..., FEATURE_INDEX["quiescent_current"]]
        shutdown_current = td["nodes"][..., FEATURE_INDEX["shutdown_current"]]
        
        use_ishut_current = torch.where(shutdown_current > 1e-9, shutdown_current, quiescent_current)
        ic_self_sleep = torch.zeros(batch_size, num_nodes, device=self.device)
        
        ic_self_sleep[is_ao & is_used] = op_current[is_ao & is_used]
        ic_self_sleep[~is_ao & is_used & parent_is_ao] = use_ishut_current[~is_ao & is_used & parent_is_ao]

        # 3. Load 암전류 소모
        load_sleep_draw_base = td["nodes"][..., FEATURE_INDEX["current_sleep"]].clone()
        load_sleep_draw = load_sleep_draw_base * always_on_nodes.float()
        load_sleep_draw[~always_on_nodes] = 0.0

        # 4. 전류 수요 전파 (LDO/Buck 효율 적용)
        current_demands_sleep = load_sleep_draw + ic_self_sleep
        
        # (B, N) 모양의 LDO/Buck 마스크를 생성합니다.
        # (B, N)
        ic_type = td["nodes"][..., FEATURE_INDEX["ic_type_idx"]]
        
        # (B, N)
        ldo_mask_b = torch.isclose(ic_type, torch.tensor(1.0, device=ic_type.device))
        
        # (B, N)
        buck_mask_b = torch.isclose(ic_type, torch.tensor(2.0, device=ic_type.device))

        # (참고: ic_mask_b_n은 이 함수에서 사용되지 않으므로 삭제하거나 (B, N)으로 만들어야 함)
        # (B, N)
        ic_mask_b_n = (self.node_type_tensor == NODE_TYPE_IC).expand(batch_size, -1)        
        
        vin = td["nodes"][..., FEATURE_INDEX["vin_min"]]
        vout = td["nodes"][..., FEATURE_INDEX["vout_min"]]
        safe_vin = torch.where(vin > 0, vin, 1e-6)
        eff_sleep = 0.35 # (고정 효율)
        
        for _ in range(num_nodes):
            i_out_sleep = (adj_matrix_T.transpose(-1, -2) @ current_demands_sleep.unsqueeze(-1)).squeeze(-1)
            
            new_demands_sleep = load_sleep_draw + ic_self_sleep
            
            # LDO: I_in = I_out
            new_demands_sleep[ldo_mask_b] += i_out_sleep[ldo_mask_b]
            
            # Buck: I_in = P_out / (Eff * V_in)
            p_out_sleep_buck = vout[buck_mask_b] * i_out_sleep[buck_mask_b]
            p_in_sleep_buck = p_out_sleep_buck / eff_sleep
            i_in_sleep_buck = p_in_sleep_buck / safe_vin[buck_mask_b]
            new_demands_sleep[buck_mask_b] += i_in_sleep_buck
            
            if torch.allclose(current_demands_sleep, new_demands_sleep, atol=1e-8):
                break
            current_demands_sleep = new_demands_sleep

        # 5. 배터리 총 암전류
        battery_children_mask = adj_matrix[:, BATTERY_NODE_IDX, :]
        total_sleep_current = (current_demands_sleep * battery_children_mask).sum(dim=1)
        
        return total_sleep_current # (B,)