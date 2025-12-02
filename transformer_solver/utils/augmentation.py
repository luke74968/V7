# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 

import torch
from tensordict import TensorDict

def augment_data(td: TensorDict, env, n_aug: int = 8) -> TensorDict:
    """
    IC 템플릿(Power IC)의 순서를 무작위로 섞어 데이터를 증강합니다.
    모델이 특정 인덱스의 IC를 선호하는 것을 방지하고, 스펙(Feature) 기반 선택을 유도합니다.
    
    Args:
        td: 원본 TensorDict (Batch_Size, ...)
        env: 환경 객체 (레이아웃 파악용)
        n_aug: 증강 배수 (예: 8이면 배치가 8배로 커짐)
    """
    if n_aug <= 1:
        return td

    batch_size = td.batch_size[0]
    num_nodes = env.N_max
    device = td.device

    # 1. 배치 확장 (B -> B * N_aug)
    aug_td = td.clone()
    aug_td = aug_td.expand(n_aug, *aug_td.batch_size).contiguous().view(-1)
    
    # 2. IC 템플릿 인덱스 범위 계산
    #    레이아웃: [Battery] + [Loads] + [Templates] + [Empty]
    num_battery = env.generator.num_battery
    num_loads = env.generator.num_loads
    num_templates = env.generator.num_templates
    
    ic_start_idx = num_battery + num_loads
    ic_end_idx = ic_start_idx + num_templates
    
    # 3. 셔플링 인덱스 생성
    #    기본적으로 [0, 1, 2, ...] 순서 유지
    base_idx = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size * n_aug, -1)
    new_idx = base_idx.clone()

    #    각 배치마다 IC 템플릿 구간만 랜덤하게 섞음
    for i in range(batch_size * n_aug):
        template_indices = torch.arange(ic_start_idx, ic_end_idx, device=device)
        shuffled_indices = template_indices[torch.randperm(num_templates, device=device)]
        new_idx[i, ic_start_idx:ic_end_idx] = shuffled_indices

    # 4. 텐서 재배열 (Gather)
    
    # (A) Nodes 피처 재배열
    d_dim = aug_td["nodes"].size(-1)
    idx_nodes = new_idx.unsqueeze(-1).expand(-1, -1, d_dim)
    aug_td["nodes"] = aug_td["nodes"].gather(1, idx_nodes)

    # (B) 행렬(Matrix) 재배열 헬퍼 함수
    def permute_matrix(matrix_tensor, perm_idx):
        # Row Shuffle
        row_idx = perm_idx.unsqueeze(-1).expand(-1, -1, num_nodes)
        shuffled_rows = matrix_tensor.gather(1, row_idx)
        # Col Shuffle
        col_idx = perm_idx.unsqueeze(1).expand(-1, num_nodes, -1)
        shuffled_cols = shuffled_rows.gather(2, col_idx)
        return shuffled_cols

    # (C) 관련 행렬들 모두 섞기
    if "connectivity_matrix" in aug_td.keys():
        aug_td["connectivity_matrix"] = permute_matrix(aug_td["connectivity_matrix"], new_idx)
        
    # (Matrix Prompt는 시퀀싱 제약(Load간 관계)이므로 IC 순서가 바뀌어도 영향 없지만,
    #  N*N 구조를 유지하기 위해 같이 섞어주는 것이 안전함)
    if "matrix_prompt_features" in aug_td.keys():
        aug_td["matrix_prompt_features"] = permute_matrix(aug_td["matrix_prompt_features"], new_idx)
        
    if "attention_mask" in aug_td.keys():
        aug_td["attention_mask"] = permute_matrix(aug_td["attention_mask"], new_idx)

    # (주의: Load 순서는 건드리지 않았으므로, Env의 Load 제약조건 인덱스는 안전함)

    return aug_td