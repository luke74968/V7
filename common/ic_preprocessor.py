#common/ic_preprocessor.py

"""
IC 전처리기 (common/ic_preprocessor.py)

이 파일은 `config_loader`가 읽어들인 '원본 IC' 목록을 받아,
각 솔버(OR-Tools, Transformer)의 요구사항에 맞게
'특화된 IC 후보 목록'을 생성하는 전처리 로직을 제공합니다.

주요 기능 (듀얼 모드 아키텍처):

[솔버별 확장 함수]
1. expand_ic_instances (OR-Tools용):
   - SAT 솔버(OR-Tools)를 위해 가능한 모든 변수를 미리 생성합니다.
   - Load 수량과 독점 제약까지 고려하여 '_copy1', '_copy2' 등 
     모든 '특화 인스턴스'를 미리 생성(Pre-Spawn)합니다.

2. expand_ic_templates (Transformer용):
   - 강화학습(Transformer)의 "Lazy Spawn" 전략을 지원합니다.
   - (Type, Vin, Vout) 조합 당 단 하나의 'IC 템플릿'만 생성합니다.
     (복제본은 `solver_env`에서 동적으로 생성됨)

[공용 헬퍼 함수]
3. calculate_derated_current_limit:
   - 1, 2번 함수가 호출하는 공용 함수입니다.
   - `original_i_limit`(원본 스펙)을 바탕으로 IC의 열(Thermal) 제약을 계산하여,
     `i_limit`(실제 유효 한계값) 필드를 채웁니다.

4. prune_dominated_ics:
   - 1, 2번 함수가 생성한 리스트(인스턴스 또는 템플릿)를 입력받습니다.
   - 다른 IC보다 모든 면에서 열등한(즉, "지배당하는") IC들을
     제거하여 최종 후보 목록을 최적화합니다.
"""

import copy
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# 정의한 데이터 클래스들을 임포트합니다.
from .data_classes import Battery, Load, PowerIC, LDO, BuckConverter

# ---
# 1. OR-Tools용: 모든 복제본 '인스턴스' 생성
# ---

def expand_ic_instances(
    available_ics: List[PowerIC], 
    loads: List[Load], 
    battery: Battery, 
    constraints: Dict[str, Any]
) -> Tuple[List[PowerIC], Dict[str, List[str]]]:
    """
    [OR-Tools용]
    모든 유효한 (Vin, Vout) 조합과 Load 수량, 독점(Exclusive) 제약을
    고려하여 `_copy1`, `_copy2`... 등 모든 '특화 인스턴스'를 미리 생성합니다.
    
    Returns:
        Tuple[List[PowerIC], Dict[str, List[str]]]:
            1. candidate_ics: 생성된 모든 '특화 인스턴스' 객체 리스트
            2. ic_groups: `_copy`로 묶인 IC 이름 그룹 
    """
    
    print("⚙️ (OR-Tools용): IC 인스턴스 확장 시작...")
    
    potential_vout = sorted(list(set(load.voltage_typical for load in loads)))
    battery.vout = (battery.voltage_min + battery.voltage_max) / 2
    potential_vin = sorted(list(set([battery.vout] + potential_vout)))
    
    candidate_ics = []
    
    # ic_groups 딕셔너리 초기화 
    ic_groups = {} 
    
    # 로직: 독점 레일용 추가 복제본 수 계산
    exclusive_loads_per_vout = defaultdict(int)
    for load in loads:
        if load.independent_rail_type in ['exclusive_path', 'exclusive_supplier']:
            exclusive_loads_per_vout[load.voltage_typical] += 1

    for template_ic in available_ics:
        for vin in potential_vin:
            for vout in potential_vout:
                
                # 전압 호환성 검사
                if not (template_ic.vin_min <= vin <= template_ic.vin_max and 
                        template_ic.vout_min <= vout <= template_ic.vout_max):
                    continue
                if isinstance(template_ic, LDO) and vin < (vout + template_ic.v_dropout):
                    continue
                if isinstance(template_ic, BuckConverter) and vin <= vout:
                    continue
                
                # 로직: 필요한 복제본 수 계산
                num_potential_loads = sum(1 for load in loads if load.voltage_typical == vout)
                extra_instances = exclusive_loads_per_vout[vout]
                num_to_create = num_potential_loads + extra_instances

                if num_to_create == 0:
                    continue

                group_key = f"{template_ic.name}@{vin:.1f}Vin_{vout:.1f}Vout"
                
                # 현재 그룹 리스트 초기화
                current_group = []
                
                for i in range(num_to_create):
                    # 템플릿 복제 및 '특화'
                    concrete_ic = copy.deepcopy(template_ic)
                    concrete_ic.vin, concrete_ic.vout = vin, vout
                    concrete_ic.name = f"{group_key}_copy{i+1}"
                    
                    # 'i_limit' (0.0)을 '열 제약 한계값'으로 덮어씀
                    derated_limit = calculate_derated_current_limit(concrete_ic, constraints)
                    
                    if derated_limit > 0:
                        concrete_ic.i_limit = derated_limit
                        candidate_ics.append(concrete_ic)
                        # 그룹에 IC 이름 추가
                        current_group.append(concrete_ic.name)
                
                # 그룹 정보 저장 
                if current_group:
                    ic_groups[group_key] = current_group
                        
    print(f"   - (OR-Tools용): 생성된 특화 IC 인스턴스 (Pruning 전): {len(candidate_ics)}개")
    
    # ic_groups 딕셔너리 반환
    return candidate_ics, ic_groups

# ---
# 2. Transformer용: '템플릿' 생성 (Lazy Spawn)
# ---

def expand_ic_templates(
    available_ics: List[PowerIC], 
    loads: List[Load], 
    battery: Battery, 
    constraints: Dict[str, Any]
) -> List[PowerIC]:
    """
    (Vin, Vout) 조합이 유효한 'IC 템플릿'을 생성합니다.
    `_copy` (복제본)를 미리 생성하지 않습니다. (Lazy Spawn)
    """
    
    print("⚙️ Transformer용: IC 템플릿 생성 (Lazy Spawn 방식) 시작...")
    
    potential_vout = sorted(list(set(load.voltage_typical for load in loads)))
    battery.vout = (battery.voltage_min + battery.voltage_max) / 2
    potential_vin = sorted(list(set([battery.vout] + potential_vout)))
    
    template_ics = {} # (Type, Vin, Vout) 키로 중복 방지

    for template_ic in available_ics:
        for vin in potential_vin:
            for vout in potential_vout:
                
                # 전압 호환성 검사
                if not (template_ic.vin_min <= vin <= template_ic.vin_max and 
                        template_ic.vout_min <= vout <= template_ic.vout_max):
                    continue
                if isinstance(template_ic, LDO) and vin < (vout + template_ic.v_dropout):
                    continue
                if isinstance(template_ic, BuckConverter) and vin <= vout:
                    continue
                
                template_key = (template_ic.name, vin, vout)
                if template_key in template_ics:
                    continue

                concrete_template = copy.deepcopy(template_ic)
                concrete_template.vin, concrete_template.vout = vin, vout
                concrete_template.name = f"{template_ic.name}@{vin:.1f}Vin_{vout:.1f}Vout"
                
                derated_limit = calculate_derated_current_limit(concrete_template, constraints)
                
                if derated_limit > 0:
                    concrete_template.i_limit = derated_limit
                    template_ics[template_key] = concrete_template
                        
    final_templates = list(template_ics.values())
    print(f"   - Transformer용: 생성된 고유 IC 템플릿 (Pruning 전): {len(final_templates)}개")
    return final_templates


# ---
# 3. 공용 헬퍼 함수: 열 제약(Thermal) 계산
# ---

def calculate_derated_current_limit(ic: PowerIC, constraints: Dict[str, Any]) -> float:
    """
    IC의 열(Thermal) 제약조건을 고려하여 실제 사용 가능한 전류 한계(derated limit)를 계산합니다.
    PowerIC 객체 (ic.vin, ic.vout이 설정된)를 입력으로 받습니다.

    """
    ambient_temp = constraints.get('ambient_temperature', 25)
    thermal_margin_percent = constraints.get('thermal_margin_percent', 0)
    
    if ic.theta_ja == 0:
        return ic.original_i_limit

    temp_rise_allowed = ic.t_junction_max - ambient_temp
    if temp_rise_allowed <= 0:
        return 0.0
    
    p_loss_max = (temp_rise_allowed / (ic.theta_ja * (1 + thermal_margin_percent)))
    i_limit_based_temp = ic.original_i_limit
    
    if isinstance(ic, LDO):
        vin, vout = ic.vin, ic.vout
        op_current = ic.operating_current
        numerator = p_loss_max - (vin * op_current)
        denominator = vin - vout
        if denominator > 0 and numerator > 0:
            i_limit_based_temp = numerator / denominator
            
    elif isinstance(ic, BuckConverter):
        # 이진 탐색 로직 (고정 효율 0.9 사용)
        low, high = 0.0, ic.original_i_limit
        i_limit_based_temp = 0.0
        for _ in range(100):
            mid = (low + high) / 2
            if mid < 1e-6: break
            power_loss_at_mid = ic.calculate_power_loss(ic.vin, mid) # 90% 고정 효율
            if power_loss_at_mid <= p_loss_max:
                i_limit_based_temp = mid
                low = mid
            else:
                high = mid
                
    # 원본 스펙 한계와 열 제약 한계 중 *더 작은* 값을 실제 한계로 반환
    return min(ic.original_i_limit, i_limit_based_temp)

# ---
# 4. 공용 헬퍼 함수: '지배당하는' (Dominated) IC 제거
# ---

def _dominates_b_over_a(a: PowerIC, b: PowerIC) -> bool:
    """ [헬퍼] IC 'b'가 IC 'a'를 '지배'하는지(더 우수한지) 확인합니다. """
    
    # b가 a보다 좋거나 같아야 하는 스펙 (값이 낮을수록 좋음)
    if not (b.cost <= a.cost and
            b.theta_ja <= a.theta_ja and
            b.quiescent_current <= a.quiescent_current):
        return False

    # b가 a보다 좋거나 같아야 하는 스펙 (값이 높을수록 좋음)
    if not (b.i_limit >= a.i_limit and # *열 제약이 적용된* i_limit 비교
            b.t_junction_max >= a.t_junction_max):
        return False
        
    if isinstance(a, LDO):
        if not (b.v_dropout <= a.v_dropout):
            return False

    # 최소 한 가지 면에서 '엄격하게' 더 좋은지 확인
    strict_improvement = (
        (b.cost < a.cost) or
        (b.theta_ja < a.theta_ja) or
        (b.quiescent_current < a.quiescent_current) or
        (b.i_limit > a.i_limit) or
        (b.t_junction_max > a.t_junction_max) or
        (isinstance(a, LDO) and b.v_dropout < a.v_dropout)
    )
    return strict_improvement

def prune_dominated_ics(ic_list: List[PowerIC]) -> List[PowerIC]:
    """
    IC 리스트 (템플릿 또는 인스턴스)를 받아 '지배당하는' IC들을 제거합니다.
    N^2 비교 대신, (Type, Vin, Vout)이 동일한 그룹 내에서만 비교하여 효율성을 높였습니다.
    """
    print("🔪 Dominance Pruning (지배 IC 제거) 시작...")
    
    groups = defaultdict(list)
    for ic in ic_list:
        # (Type, Vin, Vout) 키로 그룹화
        key = (ic.type, ic.vin, ic.vout)
        groups[key].append(ic)

    final_ic_list = []
    
    for key, group in groups.items():
        keep = [True] * len(group)
        for i, a in enumerate(group):
            if not keep[i]: continue
            for j, b in enumerate(group):
                if i == j: continue
                if _dominates_b_over_a(a, b):
                    keep[i] = False
                    break
        
        for ic, k in zip(group, keep):
            if k:
                final_ic_list.append(ic)

    removed_count = len(ic_list) - len(final_ic_list)
    print(f"   -  {removed_count}개의 지배되는 IC 제거 완료.")
    
    return final_ic_list