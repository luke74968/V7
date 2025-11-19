# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
"""
OR-Tools CP-SAT 모델 정의 (or_tools_solver/core.py)

이 파일은 OR-Tools CP-SAT 솔버를 위한 변수와 제약 조건을 정의하는
핵심 로직을 포함합니다.
"""

import json
import copy
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from ortools.sat.python import cp_model

# data_classes를 임포트
from common.data_classes import Battery, Load, PowerIC, LDO, BuckConverter

# 정수 연산을 위한 스케일 
SCALE = 1_000_000_000


# ---
# 1. 솔버 콜백 클래스 
# ---

class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """모든 유효한 해를 수집하는 콜백"""
    def __init__(self, ic_is_used, edges):
        super().__init__()
        self.__solution_count = 0
        self.__ic_is_used = ic_is_used
        self.__edges = edges
        self.solutions = []
    def on_solution_callback(self):
        self.__solution_count += 1
        current_solution = {
            "score": self.ObjectiveValue(),
            "used_ic_names": {name for name, var in self.__ic_is_used.items() if self.Value(var)},
            "active_edges": [(p, c) for (p, c), var in self.__edges.items() if self.Value(var)]}
        self.solutions.append(current_solution)
    def solution_count(self): return self.__solution_count

class SolutionLogger(cp_model.CpSolverSolutionCallback):
    """해를 찾으면 로그를 찍고 중지하는 콜백 (대표해 탐색용)"""
    def __init__(self, ic_is_used, edges, limit=1):
        super().__init__()
        self.__solution_count = 0
        self.__ic_is_used = ic_is_used
        self.__edges = edges
        self.limit = limit
        self.solutions = []
    def on_solution_callback(self):
        if len(self.solutions) >= self.limit:
            self.StopSearch()
            return
        self.__solution_count += 1
        print(f"  -> 대표 솔루션 #{self.__solution_count} 발견!")
        current_solution = {
            "score": self.ObjectiveValue(),
            "used_ic_names": {name for name, var in self.__ic_is_used.items() if self.Value(var)},
            "active_edges": [(p, c) for (p, c), var in self.__edges.items() if self.Value(var)]
        }
        self.solutions.append(current_solution)

# ---
# 2. 모델 변수 및 제약 조건 정의
# ---

def _initialize_model_variables(model, candidate_ics, loads, battery):
    """모델의 기본 변수들(노드, 엣지, IC 사용 여부)을 생성하고 반환합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics
    all_nodes = parent_nodes + loads 
    node_names = list(set(n.name for n in all_nodes))
    ic_names = [ic.name for ic in candidate_ics]
    
    edges = {}
    for p in parent_nodes:
        for c in all_ic_and_load_nodes:
            if p.name == c.name: continue
            is_compatible = False
            if p.name == battery.name:
                # 배터리는 IC의 부모가 될 수 있음
                if isinstance(c, PowerIC) and (battery.voltage_min <= c.vin <= battery.voltage_max):
                    is_compatible = True
            elif isinstance(p, PowerIC):
                # IC는 다른 IC 또는 Load의 부모가 될 수 있음
                child_vin_req = c.vin if isinstance(c, PowerIC) else c.voltage_typical
                if p.vout == child_vin_req:
                    is_compatible = True
            if is_compatible:
                edges[(p.name, c.name)] = model.NewBoolVar(f'edge_{p.name}_to_{c.name}')
    
    ic_is_used = {ic.name: model.NewBoolVar(f'is_used_{ic.name}') for ic in candidate_ics}
    
    print(f"   - 생성된 'edge' 변수: {len(edges)}개")
    return all_nodes, parent_nodes, node_names, ic_names, edges, ic_is_used

def add_base_topology_constraints(model, candidate_ics, loads, battery, edges, ic_is_used):
    """전력망의 가장 기본적인 연결 규칙을 정의합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    parent_nodes = [battery] + candidate_ics

    # 1. 사용되는 IC는 반드시 (하나 이상의) 출력이 있어야 함
    for ic in candidate_ics:
        outgoing = [edges[ic.name, c.name] for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if outgoing:
            model.Add(sum(outgoing) > 0).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(outgoing) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())
        else:
            # 출력이 불가능한 IC는 절대 사용되지 않음
            model.Add(ic_is_used[ic.name] == False)

    # 2. 모든 부하(Load)는 반드시 하나의 부모를 가져야 함
    for load in loads:
        possible_parents = [edges[p.name, load.name] for p in parent_nodes if (p.name, load.name) in edges]
        if possible_parents: 
            model.AddExactlyOne(possible_parents)
        # else: 이 Load는 전원을 공급받을 수 없음 (오류)

    # 3. 사용되는 IC는 반드시 하나의 부모를 가져야 함
    for ic in candidate_ics:
        incoming = [edges[p.name, ic.name] for p in parent_nodes if (p.name, ic.name) in edges]
        if incoming:
            model.Add(sum(incoming) == 1).OnlyEnforceIf(ic_is_used[ic.name])
            model.Add(sum(incoming) == 0).OnlyEnforceIf(ic_is_used[ic.name].Not())

def add_ic_group_constraints(model, ic_groups, ic_is_used):
    """복제된 IC 그룹 내에서의 사용 순서를 강제합니다 (예: copy2 사용시 copy1도 사용)"""
    for copies in ic_groups.values():
        for i in range(len(copies) - 1):
            # copy(i+1)이 사용되면, copy(i)도 반드시 사용되어야 함
            model.AddImplication(ic_is_used[copies[i+1]], ic_is_used[copies[i]])

def add_current_limit_constraints(model, candidate_ics, loads, constraints, edges):
    """IC의 전류 한계(열 마진, 전기 마진) 제약 조건을 추가합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    
    # 1. 각 자식 노드가 부모로부터 얼마나 많은 전류를 끌어당기는지 추정
    child_current_draw = {node.name: int(node.current_active * SCALE) for node in loads}
    
    # IC가 자식일 경우, IC의 입력 전류를 추정 
    potential_loads_for_ic = defaultdict(list)
    for ic in candidate_ics:
        for load in loads:
            if ic.vout == load.voltage_typical:
                potential_loads_for_ic[ic.name].append(load.current_active)
                
    for ic in candidate_ics:
        # 이 IC가 공급할 수 있는 최대 전류 (열 제약 적용된 i_limit)
        max_potential_i_out = sum(potential_loads_for_ic[ic.name])
        realistic_i_out = min(ic.i_limit, max_potential_i_out) 
        i_in_active = ic.calculate_active_input_current(vin=ic.vin, i_out=realistic_i_out)
        child_current_draw[ic.name] = int(i_in_active * SCALE)

    current_margin = constraints.get('current_margin', 0.1)
    
    for p in candidate_ics: # 부모(Parent) IC
        # 2. 부모 IC의 총 출력 전류 = 연결된 자식들의 전류 요구량 합
        terms = [child_current_draw[c.name] * edges[p.name, c.name] 
                 for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        
        if terms:
            # 3. 제약 조건 추가
            # p.i_limit은 '열 제약 한계'
            model.Add(sum(terms) <= int(p.i_limit * SCALE))
            # p.original_i_limit은 '전기 스펙 한계' (마진 적용)
            model.Add(sum(terms) <= int(p.original_i_limit * (1 - current_margin) * SCALE))

def add_power_sequence_constraints(model, candidate_ics, loads, battery, constraints, node_names, edges, ic_is_used):
    """ 정수 '스테이지' 변수를 사용하여 전원 시퀀스 제약 조건을 추가합니다."""

    if 'power_sequences' not in constraints or not constraints['power_sequences']:
        return

    num_nodes = len(node_names)
    stage = {name: model.NewIntVar(0, num_nodes - 1, f"stage_{name}") for name in node_names}
    model.Add(stage[battery.name] == 0)

    for (p_name, c_name), edge_var in edges.items():
        model.Add(stage[c_name] >= stage[p_name] + 1).OnlyEnforceIf(edge_var)

    for seq in constraints['power_sequences']:
        if seq.get('f') != 1: continue
        j_name, k_name = seq['j'], seq['k']
        if j_name not in node_names or k_name not in node_names: continue

        j_parents = [(p.name, edges[p.name, j_name]) for p in candidate_ics if (p.name, j_name) in edges]
        k_parents = [(p.name, edges[p.name, k_name]) for p in candidate_ics if (p.name, k_name) in edges]
        if not j_parents or not k_parents: continue

        j_parent_stage = model.NewIntVar(0, num_nodes - 1, f"stage_parent_of_{j_name}")
        k_parent_stage = model.NewIntVar(0, num_nodes - 1, f"stage_parent_of_{k_name}")
        
        for p_name, edge_var in j_parents:
            model.Add(j_parent_stage == stage[p_name]).OnlyEnforceIf(edge_var)
        for p_name, edge_var in k_parents:
            model.Add(k_parent_stage == stage[p_name]).OnlyEnforceIf(edge_var)
        
        # 핵심 제약: k 부모의 스테이지가 j 부모의 스테이지보다 커야 한다
        model.Add(k_parent_stage > j_parent_stage)

        # 동일 부모 금지 (f=1일 때)
        for p_ic_name, j_edge_var in j_parents:
            for q_ic_name, k_edge_var in k_parents:
                if p_ic_name == q_ic_name:
                    model.AddBoolOr([j_edge_var.Not(), k_edge_var.Not()])   

def add_independent_rail_constraints(model, loads, candidate_ics, all_nodes, parent_nodes, edges):
    """독립 레일(Independent Rail) 제약 조건을 모델에 추가합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    
    num_children_all = {p.name: model.NewIntVar(0, len(all_ic_and_load_nodes), f"num_children_all_{p.name}") for p in parent_nodes}
    for p in parent_nodes:
        outgoing_edges = [edges[p.name, c.name] for c in all_ic_and_load_nodes if (p.name, c.name) in edges]
        model.Add(num_children_all[p.name] == sum(outgoing_edges))

    for load in loads:
        rail_type = load.independent_rail_type
        if rail_type == 'exclusive_supplier':
            for p_ic in candidate_ics:
                if (p_ic.name, load.name) in edges:
                    model.Add(num_children_all[p_ic.name] == 1).OnlyEnforceIf(edges[(p_ic.name, load.name)])
        
        elif rail_type == 'exclusive_path':
            is_on_exclusive_path = {node.name: model.NewBoolVar(f"on_exc_path_{load.name}_{node.name}") for node in all_nodes}
            model.Add(is_on_exclusive_path[load.name] == 1)
            for other_load in loads:
                if other_load.name != load.name:
                    model.Add(is_on_exclusive_path[other_load.name] == 0)
            
            for c_node in all_ic_and_load_nodes:
                for p_node in parent_nodes:
                    if (p_node.name, c_node.name) in edges:
                        model.AddImplication(is_on_exclusive_path[c_node.name], is_on_exclusive_path[p_node.name]).OnlyEnforceIf(edges[(p_node.name, c_node.name)])
            
            for p_ic in candidate_ics:
                model.Add(num_children_all[p_ic.name] <= 1).OnlyEnforceIf(is_on_exclusive_path[p_ic.name])


def add_always_on_constraints(model, all_nodes, loads, candidate_ics, battery, edges):
    """Always-On 경로를 추적하는 제약 조건을 추가합니다."""
    all_ic_and_load_nodes = candidate_ics + loads
    is_always_on_path = {node.name: model.NewBoolVar(f"is_ao_{node.name}") for node in all_nodes}
    
    model.Add(is_always_on_path[battery.name] == 1) # 배터리는 항상 AO

    for ld in loads:
        model.Add(is_always_on_path[ld.name] == int(ld.always_on_in_sleep))
        
    for ic in candidate_ics:
        children = [c for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        if not children:
            model.Add(is_always_on_path[ic.name] == 0)
            continue
        
        # z = (edge(ic->ch) AND is_always_on_path(ch))
        z_list = []
        for ch in children:
            e = edges[(ic.name, ch.name)]
            z = model.NewBoolVar(f"ao_and_{ic.name}__{ch.name}")
            model.Add(z <= e); model.Add(z <= is_always_on_path[ch.name]); model.Add(z >= e + is_always_on_path[ch.name] - 1)
            z_list.append(z)
            
        # ic.is_ao = OR(z_list)
        for z in z_list: model.Add(is_always_on_path[ic.name] >= z)
        model.Add(is_always_on_path[ic.name] <= sum(z_list))
        
    return is_always_on_path


def add_sleep_current_constraints(model, battery, candidate_ics, loads, constraints, edges, is_always_on_path):
    """암전류(Sleep Current) 제약 조건을 추가합니다."""
   
    max_sleep = constraints.get('max_sleep_current', 0.0)
    if max_sleep <= 0:
        return # 암전류 제약 없음

    # 헬퍼 함수 
    def bool_and(a, b, name):
        w = model.NewBoolVar(name)
        model.Add(w <= a); model.Add(w <= b); model.Add(w >= a + b - 1)
        return w
    def gate_const_by_bool(const_int, b, name):
        y = model.NewIntVar(0, max(0, const_int), name)
        model.Add(y == const_int).OnlyEnforceIf(b); model.Add(y == 0).OnlyEnforceIf(b.Not())
        return y
    def gate_int_by_bool(x, ub, b, name):
        y = model.NewIntVar(0, max(0, ub), name)
        model.Add(y == x).OnlyEnforceIf(b); model.Add(y == 0).OnlyEnforceIf(b.Not())
        return y

    parent_nodes = [battery] + candidate_ics
    all_ic_and_load_nodes = candidate_ics + loads
    
    # 최대 전류량 계산 
    max_ic_self_current = sum(
        int(max(ic.operating_current, ic.quiescent_current, ic.shutdown_current or 0) * SCALE)
        for ic in candidate_ics
    )
    NODE_UB = max_ic_self_current + sum(int(ld.current_sleep * SCALE) for ld in loads) + 1

    node_sleep_in = {}
    node_sleep_ub = {}

    for ld in loads:
        const_val = max(0, int(ld.current_sleep * SCALE))
        v = gate_const_by_bool(const_val, is_always_on_path[ld.name], f"sleep_in_{ld.name}")
        node_sleep_in[ld.name] = v
        node_sleep_ub[ld.name] = const_val

    for ic in candidate_ics:
        node_sleep_in[ic.name] = model.NewIntVar(0, NODE_UB, f"sleep_in_{ic.name}")
        node_sleep_ub[ic.name] = NODE_UB

    # IC별 3-state 제약 조건 
    for ic in candidate_ics:
        # data_classes에서 스펙 읽기
        iop = max(0, int(ic.operating_current * SCALE))
        
        # I_shut이 없으면 Iq 사용
        if ic.shutdown_current is not None and ic.shutdown_current > 0:
            i_shut = max(0, int(ic.shutdown_current * SCALE))
        else:
            i_shut = max(0, int(ic.quiescent_current * SCALE))

        ic_self = model.NewIntVar(0, max(iop, i_shut), f"sleep_self_{ic.name}")
        is_ao = is_always_on_path[ic.name]

        # (A) IC의 3가지 상태(is_ao, use_ishut, no_current) 정의
        parent_is_ao = model.NewBoolVar(f"parent_of_{ic.name}_is_ao")
        possible_parents = [p for p in parent_nodes if (p.name, ic.name) in edges]
        z_list = []
        if possible_parents:
            for p in possible_parents:
                is_p_ao = is_always_on_path[p.name]
                z = bool_and(edges[(p.name, ic.name)], is_p_ao, f"z_{p.name}_{ic.name}")
                z_list.append(z)
            model.AddBoolOr([parent_is_ao.Not()] + z_list)
            for z in z_list:
                model.AddImplication(z, parent_is_ao)
        else:
            model.Add(parent_is_ao == 0)

        use_ishut = bool_and(is_ao.Not(), parent_is_ao, f"use_ishut_{ic.name}")
        no_current = bool_and(is_ao.Not(), parent_is_ao.Not(), f"no_current_{ic.name}")
        model.AddExactlyOne([is_ao, use_ishut, no_current])

        # (B) 상태에 따른 IC 자체 소모 전류(ic_self) 할당
        model.Add(ic_self == iop).OnlyEnforceIf(is_ao)
        model.Add(ic_self == i_shut).OnlyEnforceIf(use_ishut)
        model.Add(ic_self == 0).OnlyEnforceIf(no_current)

        # (C) 자식 노드들이 요구하는 전류 합산 (AO 경로 자식만)
        children = [c for c in all_ic_and_load_nodes if (ic.name, c.name) in edges]
        child_terms = []
        ub_sum = 0
        for c in children:
            # 엣지가 활성화된 자식의 sleep_in을 더함
            edge_ic_c = edges[(ic.name, c.name)]
            use_c_sleep = bool_and(edge_ic_c, is_always_on_path[c.name], f"sleep_edge_{ic.name}__{c.name}")
            ub_c = node_sleep_ub[c.name]
            term = gate_int_by_bool(node_sleep_in[c.name], ub_c, use_c_sleep, f"sleep_term_{ic.name}__{c.name}")
            child_terms.append(term)
            ub_sum += ub_c

        children_out = model.NewIntVar(0, max(0, ub_sum), f"sleep_out_{ic.name}")
        model.Add(children_out == (sum(child_terms) if child_terms else 0))

        # (D) 출력 전류를 입력 전류로 변환 (LDO/Buck)
        in_for_children = model.NewIntVar(0, NODE_UB, f"sleep_children_in_{ic.name}")
        if isinstance(ic, LDO):
            model.Add(in_for_children == children_out)
        elif isinstance(ic, BuckConverter):
            # 고정 효율 (35%) 사용 
            eff_sleep = 0.35
            vin_ref = ic.vin if ic.vin > 0 else battery.voltage_min
            vin_eff = max(1e-6, vin_ref * eff_sleep)
            # 정수 연산을 위한 스케일링 
            p = max(1, int(round(ic.vout * 1000)))
            q = max(1, int(round(vin_eff * 1000)))
            model.Add(in_for_children * q >= children_out * p)
        else:
            model.Add(in_for_children == 0)

        # (E) IC의 총 입력 전류 = 자체 소모 + 자식 공급용
        model.Add(node_sleep_in[ic.name] == ic_self + in_for_children)

    # --- 최종 제약 조건: 배터리 관점 ---
    top_children = [c for c in all_ic_and_load_nodes if (battery.name, c.name) in edges]
    final_terms = []
    for c in top_children:
        term = gate_int_by_bool(node_sleep_in[c.name], node_sleep_ub[c.name], edges[(battery.name, c.name)], f"top_term_{c.name}")
        final_terms.append(term)

    model.Add(sum(final_terms) <= int(max_sleep * SCALE))

# ---
# 3. 메인 모델 생성 함수
# ---
def create_solver_model(candidate_ics, loads, battery, constraints, ic_groups):
    """
    OR-Tools 모델을 생성하고 모든 제약 조건을 추가한 뒤 반환합니다.
    """
    print("\n🧠 OR-Tools 모델 생성 시작...")
    model = cp_model.CpModel()

    # 1. 변수 초기화
    all_nodes, parent_nodes, node_names, ic_names, edges, ic_is_used = _initialize_model_variables(
        model, candidate_ics, loads, battery
    )
    
    # 2. 제약 조건 추가
    add_base_topology_constraints(model, candidate_ics, loads, battery, edges, ic_is_used)
    add_ic_group_constraints(model, ic_groups, ic_is_used)
    add_current_limit_constraints(model, candidate_ics, loads, constraints, edges)
    add_power_sequence_constraints(model, candidate_ics, loads, battery, constraints, node_names, edges, ic_is_used)
    add_independent_rail_constraints(model, loads, candidate_ics, all_nodes, parent_nodes, edges)
    is_always_on_path = add_always_on_constraints(model, all_nodes, loads, candidate_ics, battery, edges)
    add_sleep_current_constraints(model, battery, candidate_ics, loads, constraints, edges, is_always_on_path)

    # 3. 목표 함수 설정 (비용 최소화)
    cost_objective = sum(int(ic.cost * 10000) * ic_is_used[ic.name] for ic in candidate_ics)
    model.Minimize(cost_objective)
    
    print("✅ OR-Tools 모델 생성 완료!")
    return model, edges, ic_is_used

# ---
# 4. 병렬해 탐색 함수
# ---
def find_all_load_distributions(base_solution, candidate_ics, loads, battery, constraints, viz_func, check_func):
    """
    대표 해를 기반으로, exclusive 제약조건을 위반하지 않으면서
    부하를 재분배하여 가능한 모든 유효한 병렬해를 탐색합니다.
    """
    search_settings = constraints.get('parallel_search_settings', {})
    if not search_settings.get('enabled', False):
        print("\n👑 --- 병렬 해 탐색 비활성화됨 --- 👑")
        if check_func(base_solution, candidate_ics, loads, battery, constraints):
            viz_func(base_solution, candidate_ics, loads, battery, constraints, solution_index=1)
        return

    print("\n\n👑 --- 최종 단계: 모든 부하 분배 조합 탐색 --- 👑")
    max_solutions = search_settings.get('max_solutions_to_generate', 500)

    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    child_to_parent = {c: p for p, c in base_solution['active_edges']}
    parent_to_children = defaultdict(list)
    for p, c in base_solution['active_edges']:
        parent_to_children[p].append(c)

    # Exclusive 제약에 걸린 노드 식별
    exclusive_ics = set()
    exclusive_loads = set()
    for load in loads:
        if load.independent_rail_type == 'exclusive_path':
            current_node_name = load.name
            exclusive_loads.add(current_node_name)
            while current_node_name in child_to_parent:
                parent_name = child_to_parent[current_node_name]
                if parent_name == battery.name: break
                exclusive_ics.add(parent_name)
                current_node_name = parent_name
        elif load.independent_rail_type == 'exclusive_supplier':
            parent_name = child_to_parent.get(load.name)
            if parent_name and parent_name in candidate_ics_map:
                exclusive_loads.add(load.name)
                exclusive_ics.add(parent_name)
    
    # 재분배 대상 그룹 찾기
    ic_type_to_instances = defaultdict(list)
    for ic_name in base_solution['used_ic_names']:
        ic = candidate_ics_map.get(ic_name)
        if ic and ic.name not in exclusive_ics:
            ic_type = f"📦 {ic.name.split('@')[0]} ({ic.vout:.1f}Vout)"
            ic_type_to_instances[ic_type].append(ic)

    target_group = None
    for ic_type, instances in ic_type_to_instances.items():
        if len(instances) > 1:
            total_load_pool = set()
            for inst in instances:
                loads_for_inst = [c for c in parent_to_children.get(inst.name, []) if c not in exclusive_loads]
                total_load_pool.update(loads_for_inst)
            if total_load_pool:
                target_group = {'instances': [inst.name for inst in instances], 'load_pool': list(total_load_pool)}
                break

    if not target_group:
        print("\n -> 이 해답에는 생성할 병렬해가 없습니다.")
        if check_func(base_solution, candidate_ics, loads, battery, constraints):
            viz_func(base_solution, candidate_ics, loads, battery, constraints, solution_index=1)
        return

    # 파티션 생성 
    def find_partitions(items, num_bins):
        if not items:
            yield [[] for _ in range(num_bins)]
            return
        first = items[0]
        rest = items[1:]
        for p in find_partitions(rest, num_bins):
            for i in range(num_bins):
                yield p[:i] + [[first] + p[i]] + p[i+1:]
    
    valid_solutions = []
    seen_partitions = set()
    num_instances = len(target_group['instances'])
    load_pool = target_group['load_pool']
    solution_count = 0
    fixed_edges = [edge for edge in base_solution['active_edges'] if edge[0] not in target_group['instances']]

    for p in find_partitions(load_pool, num_instances):
        if solution_count >= max_solutions:
            print(f"\n⚠️ 경고: 병렬 해 조합이 너무 많아 {max_solutions}개에서 탐색을 중단합니다.")
            break
        if len(p) != num_instances: continue

        canonical_partition = tuple(sorted([tuple(sorted(sublist)) for sublist in p]))
        if canonical_partition in seen_partitions: continue
        seen_partitions.add(canonical_partition)
        
        new_edges = list(fixed_edges)
        for i, instance_name in enumerate(target_group['instances']):
            for load_name in p[i]:
                new_edges.append((instance_name, load_name))
        
        new_solution = {"used_ic_names": base_solution['used_ic_names'], "active_edges": new_edges, "cost": base_solution['cost']}
        
        if check_func(new_solution, candidate_ics, loads, battery, constraints):
            valid_solutions.append(new_solution)
        solution_count += 1
    
    if not valid_solutions and check_func(base_solution, candidate_ics, loads, battery, constraints):
        print("\n -> 생성된 병렬해가 모두 유효하지 않아, 원본 대표해를 사용합니다.")
        valid_solutions.append(base_solution)

    print(f"\n✅ 총 {len(valid_solutions)}개의 유효한 병렬해 구조를 찾았습니다.")
    for i, solution in enumerate(valid_solutions):
        print(f"\n--- [병렬해 #{i+1}] ---")
        viz_func(solution, candidate_ics, loads, battery, constraints, solution_index=i+1)