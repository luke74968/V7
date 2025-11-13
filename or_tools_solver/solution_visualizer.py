# or_tools_solver/solution_visualizer.py

"""
OR-Tools 솔루션 검증 및 시각화 (or_tools_solver/solution_visualizer.py)

이 파일은 `core.py`가 찾은 솔루션(해답)이 유효한지 검증하고,
결과를 텍스트와 Graphviz 다이어그램으로 시각화합니다.

"""

import os
from datetime import datetime
from collections import defaultdict
from graphviz import Digraph

# common 패키지에서 data_classes 임포트
from common.data_classes import Battery, Load, PowerIC, LDO, BuckConverter

def check_solution_validity(solution, candidate_ics, loads, battery, constraints):
    """
    주어진 해답이 모든 제약조건을 만족하는지 수동으로 검증하는 함수.
    """
    print("  -> 검증 중...", end="")
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    loads_map = {load.name: load for load in loads}
    parent_to_children = defaultdict(list)
    child_to_parent = {c: p for p, c in solution['active_edges']}
    for p, c in solution['active_edges']: 
        parent_to_children[p].append(c)
    
    # 1. 전류 한계 검증
    for p_name, children_names in parent_to_children.items():
        if p_name not in candidate_ics_map: continue
        
        parent_ic = candidate_ics_map[p_name]
        actual_i_out = 0
        
        for c_name in children_names:
            if c_name in loads_map: 
                actual_i_out += loads_map[c_name].current_active
            elif c_name in candidate_ics_map:
                # 자식 노드가 IC인 경우
                child_ic = candidate_ics_map[c_name]
                child_children = parent_to_children.get(c_name, [])
                
                # 자식 IC의 출력 전류 (자손 Load들의 합)
                child_i_out = 0
                for gc_name in child_children:
                     if gc_name in loads_map:
                        child_i_out += loads_map[gc_name].current_active
                     # 이 검증기는 2-depth까지만 근사 계산합니다.
                
                # 활성 전류 계산 메소드 사용
                actual_i_out += child_ic.calculate_active_input_current(child_ic.vin, child_i_out)
        
        # `data_classes`의 필드명 사용
        if actual_i_out > parent_ic.i_limit:
            print(f" -> ❌ 열-전류 한계 위반 ({p_name})")
            return False
        if actual_i_out > parent_ic.original_i_limit * (1 - constraints.get('current_margin', 0.1)):
            print(f" -> ❌ 전기적 전류 마진 위반 ({p_name})")
            return False

    # 2. Independent Rail 검증
    for load in loads:
        rail_type = load.independent_rail_type
        if not rail_type: continue
        parent_name = child_to_parent.get(load.name)
        if not parent_name: continue

        if rail_type == 'exclusive_supplier':
            if parent_name in parent_to_children and len(parent_to_children[parent_name]) > 1:
                print(f" -> ❌ Independent Rail 위반 ({parent_name}이 exclusive_supplier 규칙 위반)")
                return False
        elif rail_type == 'exclusive_path':
            current_node_name = load.name
            while current_node_name in child_to_parent:
                parent_name = child_to_parent[current_node_name]
                if parent_name == battery.name: break
                if parent_name in parent_to_children and len(parent_to_children[parent_name]) > 1:
                    print(f" -> ❌ Independent Rail 위반 ({parent_name}가 exclusive_path 규칙 위반)")
                    return False
                current_node_name = parent_name
            
    # 3. Power Sequence 검증
    def is_ancestor(ancestor_candidate, node, parent_map):
        current_node = node
        while current_node in parent_map:
            parent = parent_map[current_node]
            if parent == ancestor_candidate: return True
            current_node = parent
        return False
    
    for rule in constraints.get('power_sequences', []):
        if rule.get('f') != 1: continue
        j_name, k_name = rule['j'], rule['k']
        j_parent = child_to_parent.get(j_name)
        k_parent = child_to_parent.get(k_name)
        if not j_parent or not k_parent: continue
        if j_parent == k_parent:
            print(f" -> ❌ Power Sequence 위반 ({j_name}와 {k_name}가 동일 부모 {j_parent} 공유)")
            return False
        if is_ancestor(ancestor_candidate=k_parent, node=j_parent, parent_map=child_to_parent):
            print(f" -> ❌ Power Sequence 위반 ({k_parent}가 {j_parent}의 전원 경로 상위에 있음)")
            return False

    print(" -> ✅ 유효")
    return True

# ---
# 2. 버전 솔루션 시각화 메인 함수
# ---
def print_and_visualize_one_solution(solution, candidate_ics, loads, battery, constraints, solution_index=0):
    """
    하나의 솔루션을 콘솔에 출력하고, 다이어그램으로 시각화하여 저장합니다.
    """
    candidate_ics_map = {ic.name: ic for ic in candidate_ics}
    loads_map = {load.name: load for load in loads}
    print(f"\n{'='*20} 솔루션 (비용: ${solution['cost']:.2f}) {'='*20}")
    
    used_ic_objects = [ic for ic in candidate_ics if ic.name in solution['used_ic_names']]
    
    # 변수명 명확화 (active/sleep 분리)
    active_current_draw = {load.name: load.current_active for load in loads}
    sleep_current_draw = {load.name: load.current_sleep for load in loads}
    
    junction_temps = {}
    actual_i_ins_active, actual_i_outs_active = {}, {}
    actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep = {}, {}, {}

    processed_ics = set()
    child_to_parent = {c: p for p, c in solution['active_edges']}

    # Always-On 경로 추적
    always_on_nodes = {l.name for l in loads if l.always_on_in_sleep}
    nodes_to_process = list(always_on_nodes)
    always_on_nodes.add(battery.name) # 배터리는 항상 AO
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        if node in child_to_parent:
            parent = child_to_parent[node]
            if parent not in always_on_nodes:
                always_on_nodes.add(parent)
                nodes_to_process.append(parent)

    # Bottom-up 방식으로 전류/전력/온도 계산
    while len(processed_ics) < len(used_ic_objects):
        progress_made = False
        
        for ic in used_ic_objects: # PowerIC 객체
            if ic.name in processed_ics: 
                continue
            
            children_names = [c for p, c in solution['active_edges'] if p == ic.name]
            
            # 모든 자식 노드가 이미 처리되었는지 확인
            if all(c in loads_map or c in processed_ics for c in children_names):
                
                # --- 활성(Active) 모드 계산 ---
                total_i_out_active = sum(active_current_draw.get(c, 0) for c in children_names)
                actual_i_outs_active[ic.name] = total_i_out_active
                
                # `data_classes` 메소드 호출
                i_in_active = ic.calculate_active_input_current(vin=ic.vin, i_out=total_i_out_active)
                power_loss = ic.calculate_power_loss(vin=ic.vin, i_out=total_i_out_active)
                
                active_current_draw[ic.name] = i_in_active
                actual_i_ins_active[ic.name] = i_in_active
                ambient_temp = constraints.get('ambient_temperature', 25)
                junction_temps[ic.name] = ambient_temp + (power_loss * ic.theta_ja)
                
                # --- 절전(Sleep) 모드 계산 (리팩토링됨) ---
                parent_name = child_to_parent.get(ic.name)
                
                # 1. IC의 3-state (AO, 비-AO/부모AO, 차단) 결정
                is_ao = ic.name in always_on_nodes
                parent_is_ao = parent_name in always_on_nodes
                
                # 2. 자식들이 요구하는 총 절전 전류
                total_i_out_sleep = sum(sleep_current_draw.get(c, 0) for c in children_names)
                
                # 3. `data_classes` 헬퍼 함수 호출
                ic_self_sleep = ic.get_self_sleep_consumption(is_ao, parent_is_ao)
                i_in_for_children = ic.calculate_sleep_input_for_children(vin=ic.vin, i_out_sleep=total_i_out_sleep)
                
                # 4. IC의 총 절전 입력 전류
                i_in_sleep = ic_self_sleep + i_in_for_children
                
                # 5. 결과 저장
                actual_i_ins_sleep[ic.name] = i_in_sleep
                actual_i_outs_sleep[ic.name] = total_i_out_sleep
                ic_self_consumption_sleep[ic.name] = ic_self_sleep
                sleep_current_draw[ic.name] = i_in_sleep # 다음 부모가 계산할 수 있도록 저장

                processed_ics.add(ic.name)
                progress_made = True

        if not progress_made and len(used_ic_objects) > 0 and len(processed_ics) < len(used_ic_objects):
            print("\n⚠️ 경고: Power Tree에서 순환 참조가 발견되어 계산을 중단합니다.")
            unprocessed_ics = [ic.name for ic in used_ic_objects if ic.name not in processed_ics]
            if unprocessed_ics: print(f"         (미처리 IC: {unprocessed_ics})")
            break

    # --- 최종 집계 ---
    primary_nodes = [c_name for p_name, c_name in solution['active_edges'] if p_name == battery.name]
    total_active_current = sum(active_current_draw.get(name, 0) for name in primary_nodes)
    total_sleep_current = sum(sleep_current_draw.get(name, 0) for name in primary_nodes)
    battery_avg_voltage = (battery.voltage_min + battery.voltage_max) / 2
    total_active_power = battery_avg_voltage * total_active_current
    
    print(f"   - 시스템 전체 슬립 전류: {total_sleep_current * 1000:.4f} mA")
    print("\n--- Power Tree 구조 ---")
    
    tree_topology = defaultdict(list)
    for p, c in solution['active_edges']: 
        tree_topology[p].append(c)
        
    def format_node_name(name, show_instance_num=False):
        if name in candidate_ics_map:
            ic = candidate_ics_map[name]
            base_name = f"📦 {ic.name.split('@')[0]} ({ic.vin:.1f}Vin -> {ic.vout:.1f}Vout)"
            if show_instance_num and '_copy' in ic.name: 
                return f"{base_name} [#{ic.name.split('_copy')[-1]}]"
            return base_name
        elif name in loads_map: 
            return f"💡 {name}"
        elif name == battery.name: 
            return f"🔋 {name}"
        return name
        
    def print_instance_tree(parent_name, prefix=""):
        children = sorted(tree_topology.get(parent_name, []))
        for i, child_name in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "└── " if is_last else "├── "
            print(prefix + connector + format_node_name(child_name, show_instance_num=True))
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_instance_tree(child_name, new_prefix)
            
    print(format_node_name(battery.name))
    root_children = sorted(tree_topology.get(battery.name, []))
    for i, child_instance_name in enumerate(root_children):
        is_last = (i == len(root_children) - 1)
        connector = "└── " if is_last else "├── "
        print(connector + format_node_name(child_instance_name, show_instance_num=True))
        new_prefix = "    " if is_last else "│   "
        print_instance_tree(child_instance_name, new_prefix)
    
    # --- 시각화 함수 호출 ---
    dot_graph = visualize_tree(
        solution, candidate_ics, loads, battery, constraints,
        junction_temps, 
        actual_i_ins_active, actual_i_outs_active, 
        actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep, 
        total_active_power, total_active_current, total_sleep_current, always_on_nodes
    )
    
    # 결과 저장 경로 
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%H%M%S")
    
    output_dir = os.path.join("result_or_tools", today_str)
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = f'or_tools_solution_{solution_index}_cost_{solution["cost"]:.2f}_{timestamp_str}'
    output_filepath = os.path.join(output_dir, base_filename)    
    
    try:
        dot_graph.render(output_filepath, view=False, cleanup=True, format='png')
        print(f"\n✅ 다이어그램을 '{output_filepath}.png' 파일로 저장했습니다.")
    except Exception as e:
        print(f"\n❌ Graphviz 렌더링 실패. (설치 확인 필요): {e}")

# ---
# 3. Graphviz 시각화 함수
# ---
def visualize_tree(solution, candidate_ics, loads, battery, constraints, junction_temps, 
                   actual_i_ins_active, actual_i_outs_active, 
                   actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep,
                   total_active_power, total_active_current, total_sleep_current, always_on_nodes):
    """솔루션 시각화 함수"""
    dot = Digraph(comment=f"Power Tree - Cost ${solution['cost']:.2f}", format='png')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')

    margin_info = f"Current Margin: {constraints.get('current_margin', 0)*100:.0f}%"
    temp_info = f"Ambient Temp: {constraints.get('ambient_temperature', 25)}°C"
    dot.attr(rankdir='LR', label=f"OR-Tools Solution\n{margin_info}, {temp_info}\n\nSolution Cost: ${solution['cost']:.2f}", labelloc='t', fontname='Arial')

    max_sleep_current_target = constraints.get('max_sleep_current', 0.0)
    battery_label = (f"🔋 {battery.name}\n\n"
        f"Total Active Power: {total_active_power:.2f} W\n"
        f"Total Active Current: {total_active_current * 1000:.1f} mA\n"
        f"Target Sleep Current: <= {max_sleep_current_target * 1000000:,.1f} µA\n"
        f"Total Sleep Current: {total_sleep_current * 1000000:,.1f} µA")

    dot.node(battery.name, battery_label, shape='box', color='darkgreen', fillcolor='white')

    # 독립 조건 노드 추적
    child_to_parent = {c: p for p, c in solution['active_edges']}
    supplier_nodes, path_nodes = set(), set()
    for load in loads:
        rail_type = load.independent_rail_type
        if rail_type == 'exclusive_supplier':
            supplier_nodes.add(load.name)
            if load.name in child_to_parent: supplier_nodes.add(child_to_parent[load.name])
        elif rail_type == 'exclusive_path':
            current_node = load.name
            while current_node in child_to_parent:
                path_nodes.add(current_node)
                parent = child_to_parent[current_node]
                path_nodes.add(parent)
                if parent == battery.name: break
                current_node = parent

    used_ics_map = {ic.name: ic for ic in candidate_ics if ic.name in solution['used_ic_names']}
    
    for ic_name, ic in used_ics_map.items():
        calculated_tj = junction_temps.get(ic_name, 0)
        i_in_active = actual_i_ins_active.get(ic_name, 0)
        i_out_active = actual_i_outs_active.get(ic_name, 0)
        i_in_sleep = actual_i_ins_sleep.get(ic_name, 0)
        i_out_sleep = actual_i_outs_sleep.get(ic_name, 0)
        i_self_sleep = ic_self_consumption_sleep.get(ic_name, 0)
        
        thermal_margin = ic.t_junction_max - calculated_tj
        
        node_style = 'rounded,filled'
        if ic_name not in always_on_nodes: node_style += ',dashed'
        fill_color = 'white'
        if ic_name in path_nodes: fill_color = 'lightblue'
        elif ic_name in supplier_nodes: fill_color = 'lightyellow'
        node_color = 'blue'
        if thermal_margin < 10: node_color = 'red'
        elif thermal_margin < 25: node_color = 'orange'
        
        label = (f"📦 {ic.name.split('@')[0]}\n\n"
            f"Vin: {ic.vin:.2f}V, Vout: {ic.vout:.2f}V\n"
            f"Iin: {i_in_active*1000:.1f}mA (Active) | {i_in_sleep*1000000:,.1f}µA (Sleep)\n"
            f"Iout: {i_out_active*1000:.1f}mA (Active) | {i_out_sleep*1000000:,.1f}µA (Sleep)\n"
            f"I_self: {ic.operating_current*1000:.1f}mA (Active) | {i_self_sleep*1000000:,.1f}µA (Sleep)\n"
            f"Tj: {calculated_tj:.1f}°C (Max: {ic.t_junction_max}°C)\n"
            f"Cost: ${ic.cost:.2f}")
        dot.node(ic_name, label, color=node_color, fillcolor=fill_color, style=node_style, penwidth='3')

    sequenced_loads = set()
    if 'power_sequences' in constraints:
        for seq in constraints['power_sequences']:
            sequenced_loads.add(seq['j']); sequenced_loads.add(seq['k'])
            
    for load in loads:
        node_style = 'rounded,filled'
        if load.name not in always_on_nodes: node_style += ',dashed'
        fill_color = 'white'
        if load.name in path_nodes: fill_color = 'lightblue'
        elif load.name in supplier_nodes: fill_color = 'lightyellow'
        label = f"💡 {load.name}\nActive: {load.voltage_typical}V | {load.current_active*1000:.1f}mA\n"
        if load.current_sleep > 0: label += f"Sleep: {load.current_sleep * 1000000:,.1f}µA\n"
        conditions = []
        if load.independent_rail_type: conditions.append(f"🔒 {load.independent_rail_type}")
        if load.name in sequenced_loads: conditions.append("⛓️ Sequence")
        if conditions: label += " ".join(conditions)
        penwidth = '3' if load.always_on_in_sleep else '1'
        dot.node(load.name, label, color='dimgray', fillcolor=fill_color, style=node_style, penwidth=penwidth)
        
    for p_name, c_name in solution['active_edges']:
        dot.edge(p_name, c_name)
        
    print(f"\n🖼️  Generating diagram for solution with cost ${solution['cost']:.2f}...")
    return dot