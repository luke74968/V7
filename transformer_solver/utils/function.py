# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import torch
from torch import Tensor
from typing import Union
from tensordict import TensorDict

# (utils/common.py의 함수들은 이 파일에서 사용되지 않습니다)


def gather_by_index(src: Tensor, idx: Tensor, dim: int = 1, squeeze: bool = True) -> Tensor:
    """
    주어진 인덱스(idx)에 따라 소스 텐서(src)에서 값을 추출(gather)합니다.
    
    예:
        src (B, N, D) = 텐서
        idx (B, 1) = 인덱스
        dim = 1 (N 차원)
    
    결과: (B, D)
    
    Args:
        src (Tensor): 원본 텐서
        idx (Tensor): 추출할 인덱스 텐서
        dim (int): 인덱싱을 적용할 차원
        squeeze (bool): 결과 텐서의 차원을 축소할지 여부
        
    Returns:
        Tensor: 인덱싱된 결과 텐서
    """
    
    # 인덱스(idx) 텐서의 차원을 원본(src) 텐서의 차원과 맞추기 위해 확장
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1 # 인덱싱할 차원은 -1로 설정
    
    # (B, 1) -> (B, 1, 1) ... (src.dim() - idx.dim()) 만큼 차원 추가
    idx_expanded = idx.view(idx.shape + (1,) * (src.dim() - idx.dim()))
    # (B, 1, 1) -> (B, N, D) 형태로 확장 (단, dim=1은 -1 유지)
    idx_expanded = idx_expanded.expand(expanded_shape)
    
    # dim=1에서 idx.size(1)이 1이고 squeeze=True이면 차원 축소
    squeeze_dim = (idx.size(dim) == 1) and squeeze
    
    result = src.gather(dim, idx_expanded)
    
    return result.squeeze(dim) if squeeze_dim else result