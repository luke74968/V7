# transformer_solver/utils/common.py

import time
import logging
import math
from typing import Union

import torch
from torch import Tensor
from tensordict import TensorDict


class TimeEstimator:
    """ 훈련 시간 및 남은 시간을 예측하는 헬퍼 클래스 """

    def __init__(self, log_fn=None):
        self.log = log_fn or logging.getLogger('TimeEstimator').info
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        """ 경과 시간(h)과 남은 시간(h)을 반환합니다. """
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        if count - self.count_zero == 0:
            return elapsed_time / 3600.0, 0.0

        remain_time = elapsed_time * remain / (count - self.count_zero)
        return elapsed_time / 3600.0, remain_time / 3600.0

    def get_est_string(self, count, total):
        """ 경과 시간과 남은 시간을 문자열(h 또는 m)로 반환합니다. """
        elapsed_time_h, remain_time_h = self.get_est(count, total)

        elapsed_time_str = (
            f"{elapsed_time_h:.2f}h" if elapsed_time_h > 1.0 else f"{elapsed_time_h*60:.2f}m"
        )
        remain_time_str = (
            f"{remain_time_h:.2f}h" if remain_time_h > 1.0 else f"{remain_time_h*60:.2f}m"
        )

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        """ 남은 시간을 계산하여 로그에 출력합니다. """
        elapsed_str, remain_str = self.get_est_string(count, total)
        self.log(
            f"Epoch {count:3d}/{total:3d}: Time Est.: Elapsed[{elapsed_str}], Remain[{remain_str}]"
        )


def _batchify_single(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """ 텐서 또는 TensorDict의 첫 번째 차원(배치)을 'repeats'만큼 복제합니다. """
    s = x.shape
    # (B, ...) -> (Repeats, B, ...) -> (Repeats*B, ...)
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(x: Union[Tensor, TensorDict], shape: Union[tuple, int]) -> Union[Tensor, TensorDict]:
    """
    POMO 스타일의 병렬 탐색을 위해 데이터를 확장하는 함수.
    (예: (B=8, N, D) -> batchify(x, 128) -> (B=1024, N, D))
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def _unbatchify_single(x: Union[Tensor, TensorDict], repeats: int) -> Union[Tensor, TensorDict]:
    """ batchify의 역연산 """
    s = x.shape
    # (Repeats*B, ...) -> (Repeats, B, ...) -> (B, Repeats, ...)
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(x: Union[Tensor, TensorDict], shape: Union[tuple, int]) -> Union[Tensor, TensorDict]:
    """ 확장된 데이터를 원래 형태로 되돌리는 함수 (POMO 결과 취합용) """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _unbatchify_single(x, s) if s > 0 else x
    return x


def clip_grad_norms(param_groups, max_norm=math.inf):
    """ 
    PyTorch 옵티마이저의 파라미터 그룹에 대해 그래디언트 클리핑을 수행합니다.
    (Gradient Explosion 방지용)
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups if group['params']
    ]
    grad_norms_cpu = [g.item() for g in grad_norms]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms_cpu] if max_norm > 0 else grad_norms_cpu
    return grad_norms_cpu, grad_norms_clipped