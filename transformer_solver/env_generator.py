# transformer_solver/env_generator.py

import json
import torch
from tensordict import TensorDict
from typing import Dict, Any, List, Tuple

# --- 공용(common) 모듈 임포트 ---
from common.config_loader import load_configuration_from_file
from common.ic_preprocessor import expand_ic_templates, prune_dominated_ics
from common.data_classes import Battery, LDO, BuckConverter, Load

# --- 현재 패키지(transformer_solver) 모듈 임포트 ---
from .definitions import (
    PocatConfig, FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM,
    NODE_TYPE_PADDING, NODE_TYPE_BATTERY, NODE_TYPE_LOAD, 
    NODE_TYPE_IC, NODE_TYPE_EMPTY
)

class PocatGenerator:
    """
    설정 파일(.json)을 읽고, 모델이 학습/추론에 사용할
    '문제 텐서(Problem Tensor)'를 생성하는 클래스입니다.
    
    주요 역할:
    1. 설정 파일을 로드하고 'IC 템플릿' 목록을 전처리합니다.
    2. (N_max, FEATURE_DIM) 크기의 텐서를 생성하고,
       [BATT][LOADS][IC_TEMPLATES][EMPTY] 레이아웃에 맞게 데이터를 채웁니다.
    3. 어텐션 마스크, 연결성 행렬 등 모델에 필요한 보조 텐서를 생성합니다.
    """
    
    def __init__(self, config_file_path: str, N_max: int):
        """
        PocatGenerator를 초기화합니다.
        
        Args:
            config_file_path (str): 로드할 .json 설정 파일 경로
            N_max (int): 모델이 처리할 고정된 최대 노드 크기 (N_MAX)
        """
        
        self.N_max = N_max # (예: 500)
        
        # 1. 설정 파일 로드
        battery_obj, original_ics_obj, loads_obj, constraints_obj = \
            load_configuration_from_file(config_file_path)

        if not battery_obj or not loads_obj:
            raise ValueError(f"설정 파일 로드 실패: {config_file_path}")

        # 2. "IC 템플릿" 생성 (Lazy Spawn을 위함)
        #    (Vin, Vout) 조합별로 IC 후보를 생성합니다.
        template_ic_objs = expand_ic_templates(
            original_ics_obj, loads_obj, battery_obj, constraints_obj
        )
        
        # 3. "IC 템플릿" 목록에서 지배적인(dominated) 후보 제거
        pruned_template_objs = prune_dominated_ics(template_ic_objs)
        
        # 4. 내부용 PocatConfig 생성 (노드 이름/타입 목록 생성용)
        #    (definitions.py의 [B, L, IC] 순서를 따름)
        config_data = {
            "battery": battery_obj.__dict__,
            "available_ics": [ic.__dict__ for ic in pruned_template_objs],
            "loads": [load.__dict__ for load in loads_obj],
            "constraints": constraints_obj
        }
        self.config = PocatConfig(**config_data)

        # 5. 텐서 레이아웃 계산
        self.num_battery = 1
        self.num_loads = len(self.config.loads)
        self.num_templates = len(self.config.available_ics)
        
        # 실제 부품(BATT, LOADS, TEMPLATES)의 총 개수
        self.num_components = self.num_battery + self.num_loads + self.num_templates
        
        if self.num_components > self.N_max:
            raise ValueError(
                f"설정 파일의 부품 개수({self.num_components})가 "
                f"N_MAX ({self.N_max})를 초과합니다."
            )
            
        # IC가 스폰(Spawn)될 수 있는 빈 슬롯의 개수
        self.num_empty_slots = self.N_max - self.num_components
        
        print(f"PocatGenerator: N_max={self.N_max} | "
              f"Layout: [1 B] + [{self.num_loads} L] + "
              f"[{self.num_templates} T] + [{self.num_empty_slots} E]")

        # 6. 재사용할 기본 텐서들을 미리 계산하고 캐시합니다.
        self._base_tensors = {}
        self._tensor_cache_by_device = {}
        self._initialize_base_tensors()

    def _initialize_base_tensors(self) -> None:
        """모델에 필요한 모든 기본 텐서(노드 피처, 마스크 등)를 미리 계산합니다."""
        
        # (N_max, D)
        node_features = self._create_feature_tensor()
        # (N_max, N_max)
        connectivity_matrix = self._create_connectivity_matrix(node_features)
        # (4,), (N_max, N_max)
        scalar_prompt, matrix_prompt = self._create_prompt_tensors()
        # (N_max, N_max)
        attention_mask = self._create_attention_mask(node_features)

        # 계산된 텐서들을 (detach된 상태로) 기본 캐시에 저장
        self._base_tensors = {
            "nodes": node_features.detach(),
            "connectivity_matrix": connectivity_matrix.detach(),
            "scalar_prompt_features": scalar_prompt.detach(),
            "matrix_prompt_features": matrix_prompt.detach(),
            "attention_mask": attention_mask.detach(),
        }
        
        base_device = node_features.device
        self._tensor_cache_by_device[base_device] = self._base_tensors

    def _create_attention_mask(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        모델의 Self-Attention을 위한 마스크를 생성합니다.
        PADDING 타입을 제외한 모든 노드(BATT, LOAD, IC, EMPTY)는
        서로 상호작용(Attend)할 수 있습니다.
        """
        # (N_max,)
        node_types = node_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        
        # PADDING(0)이 아닌 모든 노드는 True
        alive_mask_1d = (node_types != NODE_TYPE_PADDING)
        
        # (N_max, N_max)
        attention_mask = alive_mask_1d.unsqueeze(1) & alive_mask_1d.unsqueeze(0)
        return attention_mask

    def _create_feature_tensor(self) -> torch.Tensor:
        """
        [BATT][LOADS][IC_TEMPLATES][EMPTY] 레이아웃으로
        (N_max, FEATURE_DIM) 크기의 노드 피처 텐서를 생성합니다.
        """
        
        features = torch.zeros(self.N_max, FEATURE_DIM)
        ambient_temp = self.config.constraints.get("ambient_temperature", 25.0)
        
        # 모든 노드의 기본 정션 온도는 주변 온도로 설정
        features[:, FEATURE_INDEX["junction_temp"]] = ambient_temp
        
        current_idx = 0
        
        # --- 1. Battery (Active) (Slot 0) ---
        b_conf = self.config.battery
        features[current_idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] = 1.0
        features[current_idx, FEATURE_INDEX["is_active"]] = 1.0 # (상태 피처)
        features[current_idx, FEATURE_INDEX["vout_min"]] = b_conf["voltage_min"]
        features[current_idx, FEATURE_INDEX["vout_max"]] = b_conf["voltage_max"]
        current_idx += 1
        
        # --- 2. Loads (Active) (Slots 1 ~ N_loads) ---
        load_start_idx = current_idx
        for i, l_conf in enumerate(self.config.loads):
            idx = load_start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] = 1.0
            features[idx, FEATURE_INDEX["is_active"]] = 1.0 # (상태 피처)
            
            # Load 요구사항 피처 설정
            features[idx, FEATURE_INDEX["vin_min"]] = l_conf["voltage_req_min"]
            features[idx, FEATURE_INDEX["vin_max"]] = l_conf["voltage_req_max"]
            features[idx, FEATURE_INDEX["current_active"]] = l_conf["current_active"]
            features[idx, FEATURE_INDEX["current_sleep"]] = l_conf["current_sleep"]
            
            rail_type = l_conf.get("independent_rail_type")
            if rail_type == "exclusive_supplier":
                features[idx, FEATURE_INDEX["independent_rail_type"]] = 1.0
            elif rail_type == "exclusive_path":
                features[idx, FEATURE_INDEX["independent_rail_type"]] = 2.0
            
            if l_conf.get("always_on_in_sleep", False):
                features[idx, FEATURE_INDEX["always_on_in_sleep"]] = 1.0
        
        current_idx += self.num_loads

        # --- 3. IC Templates (Template) (Slots N_loads+1 ~ N_components-1) ---
        template_start_idx = current_idx
        for i, ic_conf in enumerate(self.config.available_ics):
            idx = template_start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] = 1.0
            features[idx, FEATURE_INDEX["is_template"]] = 1.0 # (상태 피처)
            
            # IC 템플릿 스펙 피처 설정
            features[idx, FEATURE_INDEX["cost"]] = ic_conf.get("cost", 0.0)
            features[idx, FEATURE_INDEX["vin_min"]] = ic_conf.get("vin", 0.0)
            features[idx, FEATURE_INDEX["vin_max"]] = ic_conf.get("vin", 0.0)
            features[idx, FEATURE_INDEX["vout_min"]] = ic_conf.get("vout", 0.0)
            features[idx, FEATURE_INDEX["vout_max"]] = ic_conf.get("vout", 0.0)
            features[idx, FEATURE_INDEX["i_limit"]] = ic_conf.get("i_limit", 0.0)
            features[idx, FEATURE_INDEX["theta_ja"]] = ic_conf.get("theta_ja", 999.0)
            features[idx, FEATURE_INDEX["t_junction_max"]] = ic_conf.get("t_junction_max", 125.0)
            features[idx, FEATURE_INDEX["quiescent_current"]] = ic_conf.get("quiescent_current", 0.0)
            features[idx, FEATURE_INDEX["shutdown_current"]] = ic_conf.get("shutdown_current", 0.0)
            features[idx, FEATURE_INDEX["op_current"]] = ic_conf.get("operating_current", 0.0)
            
            ic_type = ic_conf.get("type")
            if ic_type == 'LDO':
                features[idx, FEATURE_INDEX["ic_type_idx"]] = 1.0
            elif ic_type == 'Buck':
                features[idx, FEATURE_INDEX["ic_type_idx"]] = 2.0
                
        current_idx += self.num_templates
        
        # --- 4. Empty Slots (Spawnable) (Slots N_components ~ N_max-1) ---
        empty_start_idx = self.num_components
        if self.num_empty_slots > 0:
            features[empty_start_idx:, FEATURE_INDEX["node_type"][0] + NODE_TYPE_EMPTY] = 1.0
            features[empty_start_idx:, FEATURE_INDEX["can_spawn_into"]] = 1.0 # (상태 피처)
        
        # --- 5. Node ID (0 ~ N_max-1) 정규화 ---
        for idx in range(self.N_max):
             features[idx, FEATURE_INDEX["node_id"]] = float(idx) / self.N_max
             
        return features

    def _create_connectivity_matrix(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        노드 피처(전압)를 기반으로 (N_max, N_max) 크기의
        '잠재적 연결성' 인접 행렬을 생성합니다.
        
        (상태와 관계없이, 전압만 맞으면 True)
        """
        num_nodes = self.N_max
        
        # (N_max,)
        node_types = node_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        
        # 부모가 될 수 있는 타입: Battery(1), IC(3)
        is_parent = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_BATTERY)
        # 자식이 될 수 있는 타입: Load(2), IC(3)
        is_child = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_LOAD)
        
        parent_mask = is_parent.unsqueeze(1).expand(-1, num_nodes)
        child_mask = is_child.unsqueeze(0).expand(num_nodes, -1)
        
        # 전압 호환성 계산
        parent_vout_min = node_features[:, FEATURE_INDEX["vout_min"]].unsqueeze(1)
        parent_vout_max = node_features[:, FEATURE_INDEX["vout_max"]].unsqueeze(1)
        child_vin_min = node_features[:, FEATURE_INDEX["vin_min"]].unsqueeze(0)
        child_vin_max = node_features[:, FEATURE_INDEX["vin_max"]].unsqueeze(0)
        
        voltage_compatible = (parent_vout_min <= child_vin_max) & (parent_vout_max >= child_vin_min)
        
        # adj_matrix[i, j] = True => i가 j의 부모가 될 수 있음 (i -> j)
        adj_matrix = parent_mask & child_mask & voltage_compatible
        adj_matrix.diagonal().fill_(False) # 자기 자신에게 연결 방지
        return adj_matrix.to(torch.bool)

    def _create_prompt_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        스칼라 및 행렬 프롬프트 텐서를 생성합니다.
        (행렬 프롬프트는 파워 시퀀싱 제약조건을 인코딩합니다.)
        """
        
        constraints = self.config.constraints
        
        # 1. 스칼라 프롬프트 (제약조건 값)
        scalar_prompt_list = [
            constraints.get("ambient_temperature", 25.0),
            constraints.get("max_sleep_current", 0.0),
            constraints.get("current_margin", 0.0),
            constraints.get("thermal_margin_percent", 0.0)
        ]
        scalar_prompt_features = torch.tensor(scalar_prompt_list, dtype=torch.float32)

        # 2. 행렬 프롬프트 (파워 시퀀싱)
        matrix_prompt_features = torch.zeros(self.N_max, self.N_max, dtype=torch.float32)
        
        # (definitions.py의 PocatConfig가 [B,L,IC] 순서를 보장하므로
        #  node_names 리스트의 인덱스는 텐서 레이아웃과 일치합니다)
        node_name_to_idx = {name: i for i, name in enumerate(self.config.node_names)}

        for seq in constraints.get("power_sequences", []):
            j_name, k_name = seq.get('j'), seq.get('k')
            j_idx = node_name_to_idx.get(j_name)
            k_idx = node_name_to_idx.get(k_name)
            
            # (j_idx, k_idx가 실제 부품 슬롯 내에 있는지 확인)
            if j_idx is not None and k_idx is not None and \
               j_idx < self.num_components and k_idx < self.num_components:
                
                # (예: matrix_prompt[LOAD_A, LOAD_B] = 1.0)
                matrix_prompt_features[j_idx, k_idx] = 1.0

        return scalar_prompt_features, matrix_prompt_features

    def _get_device_base_tensors(self, device: Any = None) -> Dict[str, torch.Tensor]:
        """미리 계산된 텐서를 요청된 디바이스로 이동시키고 캐시합니다."""
        if device is None:
            # 기본 디바이스의 텐서 사용
            return next(iter(self._tensor_cache_by_device.values()))

        device = torch.device(device)
        if device in self._tensor_cache_by_device:
            # 캐시된 텐서 반환
            return self._tensor_cache_by_device[device]

        base_device, base_tensors = next(iter(self._tensor_cache_by_device.items()))
        if device == base_device:
            return base_tensors

        # 새 디바이스로 텐서 이동 및 캐시
        device_tensors = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in base_tensors.items()
        }
        self._tensor_cache_by_device[device] = device_tensors
        return device_tensors


    def __call__(self, batch_size: int, **kwargs) -> TensorDict:
        """
        Generator를 호출하여, 미리 계산된 텐서들을
        요청된 배치 크기(batch_size)만큼 복제하여 TensorDict로 반환합니다.
        """
        device = kwargs.get("device", None)
        base_tensors = self._get_device_base_tensors(device)

        # (B, N_max, D)
        nodes = base_tensors["nodes"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        # (B, 4)
        scalar_prompt = base_tensors["scalar_prompt_features"].detach().unsqueeze(0).expand(batch_size, -1).clone()
        # (B, N_max, N_max)
        matrix_prompt = base_tensors["matrix_prompt_features"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        # (B, N_max, N_max)
        connectivity = base_tensors["connectivity_matrix"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        # (B, N_max, N_max)
        attention_mask = base_tensors["attention_mask"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        return TensorDict({
            "nodes": nodes,
            "scalar_prompt_features": scalar_prompt,
            "matrix_prompt_features": matrix_prompt,
            "connectivity_matrix": connectivity,
            "attention_mask": attention_mask, # 모델 어텐션용 마스크
        }, batch_size=[batch_size])