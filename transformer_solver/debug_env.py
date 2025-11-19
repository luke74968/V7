# transformer_solver/debug_env.py

import torch
import argparse
import sys
import os
import pprint # (딕셔너리 출력을 위해)
from typing import Dict, List
from collections import defaultdict, Counter # [추가]
from datetime import datetime # [추가]
from graphviz import Digraph # [추가]

# (common을 참조하므로, 프로젝트 루트 경로를 sys.path에 추가)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_solver.solver_env import PocatEnv, BATTERY_NODE_IDX, PENALTY_WEIGHT_SLEEP
from transformer_solver.definitions import (
    FEATURE_INDEX, NODE_TYPE_LOAD, NODE_TYPE_IC, NODE_TYPE_EMPTY
)

from common.data_classes import LDO, BuckConverter

def get_node_name(idx: int, node_names: List[str]) -> str:
    """ 인덱스에 해당하는 노드 이름을 안전하게 반환합니다. """
    if 0 <= idx < len(node_names):
        name = node_names[idx]
        if name:
            return name
        return node_names[idx]
    if idx == -1:
        return "N/A"
    return f"SPAWNED_IC (idx:{idx})"

def visualize_debug_result(env: PocatEnv, final_td, cost: float, sleep_current: float):
    """
    디버그 결과(최종 상태)를 상세 물리량(전류, 온도 등)과 함께 시각화하여 저장합니다.
    """
    print("\n🖼️ Generating detailed debug visualization...")
    
    result_dir = "result_debug"
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. 기본 정보 준비
    node_names = env.generator.config.node_names
    loads_map = {load['name']: load for load in env.generator.config.loads}
    candidate_ics_map = {ic['name']: ic for ic in env.generator.config.available_ics}
    battery_conf = env.generator.config.battery
    constraints = env.generator.config.constraints
    
    # 텐서 추출 (Batch=1 가정)
    all_nodes_features = final_td["nodes"].squeeze(0)
    is_active_mask = final_td["is_active_mask"].squeeze(0)
    adj_matrix = final_td["adj_matrix"].squeeze(0)
    node_types = all_nodes_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)

    # 2. 동적 노드 이름 생성 (Spawn된 IC용)
    dynamic_node_names = list(node_names)
    if len(dynamic_node_names) < env.N_max:
        dynamic_node_names.extend([None] * (env.N_max - len(dynamic_node_names)))

    spawn_name_counter = Counter()
    for idx in range(len(node_names), env.N_max):
        if idx >= len(is_active_mask) or not is_active_mask[idx]: continue

        node_feat = all_nodes_features[idx]
        node_id_val = node_feat[FEATURE_INDEX["node_id"]].item()
        template_idx = int(round(node_id_val * env.N_max))

        if 0 <= template_idx < len(node_names):
            base_name = node_names[template_idx]
        else:
            base_name = f"Template_{template_idx}"

        spawn_name_counter[base_name] += 1
        dynamic_node_names[idx] = f"{base_name}#{spawn_name_counter[base_name]}"

    def get_node_name_safe(idx):
        if 0 <= idx < len(dynamic_node_names) and dynamic_node_names[idx]:
            return dynamic_node_names[idx]
        return f"Node_{idx}"

    # 3. 트리 구조 복원
    child_to_parent = {}
    parent_to_children = defaultdict(list)
    
    parent_indices, child_indices = adj_matrix.nonzero(as_tuple=True)
    for p_idx, c_idx in zip(parent_indices, child_indices):
        p_name = get_node_name_safe(p_idx.item())
        c_name = get_node_name_safe(c_idx.item())
        child_to_parent[c_name] = p_name
        parent_to_children[p_name].append(c_name)

    # 4. 상세 물리량(전류, 온도) 계산 (Bottom-up)
    junction_temps, actual_i_ins_active, actual_i_outs_active = {}, {}, {}
    actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep = {}, {}, {}
    
    # 초기값: Load들의 전류 소모량
    active_current_draw = {name: conf["current_active"] for name, conf in loads_map.items()}
    sleep_current_draw = {name: conf["current_sleep"] for name, conf in loads_map.items()}

    # Always-On, Rail 정보 추적
    always_on_nodes = {name for name, conf in loads_map.items() if conf.get("always_on_in_sleep", False)}
    always_on_nodes.add(battery_conf['name'])
    
    supplier_nodes = set()
    path_nodes = set()
    
    # AO 경로 확산
    queue = list(always_on_nodes)
    while queue:
        node = queue.pop(0)
        if node in child_to_parent:
            parent = child_to_parent[node]
            if parent not in always_on_nodes:
                always_on_nodes.add(parent)
                queue.append(parent)

    # Independent Rail 추적
    for name, conf in loads_map.items():
        rail_type = conf.get("independent_rail_type")
        if rail_type == 'exclusive_supplier':
            supplier_nodes.add(name)
            if name in child_to_parent: supplier_nodes.add(child_to_parent[name])
        elif rail_type == 'exclusive_path':
            curr = name
            while curr in child_to_parent:
                path_nodes.add(curr)
                parent = child_to_parent[curr]
                path_nodes.add(parent)
                if parent == battery_conf['name']: break
                curr = parent

    # IC 계산 루프
    active_indices = torch.where(is_active_mask)[0].tolist()
    active_ics_indices = [idx for idx in active_indices if node_types[idx] == NODE_TYPE_IC]
    processed_ics = set()

    while len(processed_ics) < len(active_ics_indices):
        progress_made = False
        for ic_idx in active_ics_indices:
            ic_name = get_node_name_safe(ic_idx)
            if ic_name in processed_ics: continue
            
            children = parent_to_children.get(ic_name, [])
            
            # 모든 자식의 전류 계산이 완료되었는지 확인
            if all(c in active_current_draw for c in children):
                
                # IC 객체 생성 (스펙 로드)
                if ic_name in candidate_ics_map:
                    ic_data = candidate_ics_map[ic_name].copy()
                    ic_type = ic_data['type']
                else:
                    # 템플릿 정보 복원
                    feat = all_nodes_features[ic_idx]
                    ic_type_idx = feat[FEATURE_INDEX["ic_type_idx"]].item()
                    ic_type = 'LDO' if ic_type_idx == 1.0 else 'Buck'
                    ic_data = {
                        'type': ic_type, 'name': ic_name,
                        'vin': feat[FEATURE_INDEX["vin_min"]].item(),
                        'vout': feat[FEATURE_INDEX["vout_min"]].item(),
                        'vin_min': feat[FEATURE_INDEX["vin_min"]].item(),
                        'vin_max': feat[FEATURE_INDEX["vin_max"]].item(),
                        'vout_min': feat[FEATURE_INDEX["vout_min"]].item(),
                        'vout_max': feat[FEATURE_INDEX["vout_max"]].item(),
                        'original_i_limit': feat[FEATURE_INDEX["i_limit"]].item(),
                        'i_limit': feat[FEATURE_INDEX["i_limit"]].item(),
                        'operating_current': feat[FEATURE_INDEX["op_current"]].item(),
                        'quiescent_current': feat[FEATURE_INDEX["quiescent_current"]].item(),
                        'shutdown_current': feat[FEATURE_INDEX["shutdown_current"]].item(),
                        'cost': feat[FEATURE_INDEX["cost"]].item(),
                        'theta_ja': feat[FEATURE_INDEX["theta_ja"]].item(),
                        't_junction_max': feat[FEATURE_INDEX["t_junction_max"]].item(),
                    }
                    if ic_type == 'LDO': ic_data['v_dropout'] = 0.0

                ic_obj = LDO(**ic_data) if ic_type == 'LDO' else BuckConverter(**ic_data)

                # --- Active 모드 계산 ---
                total_i_out_active = sum(active_current_draw[c] for c in children)
                actual_i_outs_active[ic_name] = total_i_out_active
                
                i_in_active = ic_obj.calculate_active_input_current(ic_obj.vin, total_i_out_active)
                power_loss = ic_obj.calculate_power_loss(ic_obj.vin, total_i_out_active)
                
                active_current_draw[ic_name] = i_in_active
                actual_i_ins_active[ic_name] = i_in_active
                
                # 온도 계산
                ambient_temp = constraints.get('ambient_temperature', 25.0)
                junction_temps[ic_name] = ambient_temp + (power_loss * ic_obj.theta_ja)

                # --- Sleep 모드 계산 ---
                parent_name = child_to_parent.get(ic_name)
                is_ao = ic_name in always_on_nodes
                parent_is_ao = (parent_name in always_on_nodes) or (parent_name == battery_conf['name'])

                total_i_out_sleep = sum(sleep_current_draw.get(c, 0.0) for c in children)
                
                ic_self_sleep = ic_obj.get_self_sleep_consumption(is_ao, parent_is_ao)
                i_in_for_children = ic_obj.calculate_sleep_input_for_children(ic_obj.vin, total_i_out_sleep)
                
                total_i_in_sleep = ic_self_sleep + i_in_for_children
                
                actual_i_outs_sleep[ic_name] = total_i_out_sleep
                ic_self_consumption_sleep[ic_name] = ic_self_sleep
                actual_i_ins_sleep[ic_name] = total_i_in_sleep
                sleep_current_draw[ic_name] = total_i_in_sleep

                processed_ics.add(ic_name)
                progress_made = True
        
        if not progress_made and len(processed_ics) < len(active_ics_indices):
            print("⚠️ Warning: Loop in power tree or unconnected parts detected.")
            break

    # 5. 최종 배터리 전력 계산
    primary_nodes = parent_to_children.get(battery_conf['name'], [])
    total_active_current = sum(active_current_draw.get(c, 0) for c in primary_nodes)
    total_sleep_current_calc = sum(sleep_current_draw.get(c, 0) for c in primary_nodes)
    avg_batt_v = (battery_conf['voltage_min'] + battery_conf['voltage_max']) / 2
    total_active_power = avg_batt_v * total_active_current

    # 6. Graphviz 그리기
    dot = Digraph(comment=f"Debug Tree - Cost ${cost:.2f}")
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    label_text = (f"Debug Solution\nCost: ${cost:.2f}\n"
                  f"Sleep Current: {sleep_current * 1e6:.1f} uA (Target: {constraints.get('max_sleep_current', 0)*1e6:.0f} uA)")
    dot.attr(rankdir='LR', label=label_text, labelloc='t')

    # 배터리 노드
    batt_label = (f"🔋 {battery_conf['name']}\n"
                  f"Active P: {total_active_power:.2f} W\n"
                  f"Active I: {total_active_current*1000:.1f} mA\n"
                  f"Sleep I: {sleep_current*1e6:.1f} uA"
                  f"Sleep I: {total_sleep_current_calc*1e6:.1f} uA") # [수정]
    dot.node(battery_conf['name'], batt_label, shape='box', color='darkgreen', fillcolor='white')

    # 모든 노드 그리기
    for idx in active_indices:
        name = get_node_name_safe(idx)
        if name == battery_conf['name']: continue

        # 스타일
        node_style = 'rounded,filled'
        if name not in always_on_nodes: node_style += ',dashed'
        
        # Load
        if name in loads_map:
            conf = loads_map[name]
            label = f"💡 {name}\n{conf['voltage_typical']}V | {conf['current_active']*1000:.1f}mA"
            if conf['current_sleep'] > 0: label += f"\nSleep: {conf['current_sleep']*1e6:.1f}uA"
            
            fill_color = 'white'
            if name in path_nodes: fill_color = 'lightblue'
            elif name in supplier_nodes: fill_color = 'lightyellow'
            
            dot.node(name, label, color='dimgray', fillcolor=fill_color, style=node_style)
        
        # IC
        elif node_types[idx] == NODE_TYPE_IC:
            # 계산된 값 가져오기
            i_in_act = actual_i_ins_active.get(name, 0)
            i_out_act = actual_i_outs_active.get(name, 0)
            i_in_slp = actual_i_ins_sleep.get(name, 0)
            i_self_slp = ic_self_consumption_sleep.get(name, 0)
            tj = junction_temps.get(name, 0)
            
            # 템플릿 기본 정보
            feat = all_nodes_features[idx]
            vin = feat[FEATURE_INDEX["vin_min"]].item()
            vout = feat[FEATURE_INDEX["vout_min"]].item()
            cost_ic = feat[FEATURE_INDEX["cost"]].item()
            tj_max = feat[FEATURE_INDEX["t_junction_max"]].item()

            label = (f"📦 {name.split('#')[0]}\n"
                     f"Vin:{vin:.1f}V -> Vout:{vout:.1f}V\n"
                     f"Iin: {i_in_act*1000:.1f}mA (Act) | {i_in_slp*1e6:.1f}uA (Slp)\n"
                     f"Iout: {i_out_act*1000:.1f}mA (Act)\n"
                     f"I_self(Slp): {i_self_slp*1e6:.1f}uA\n"
                     f"Tj: {tj:.1f}°C (Max {tj_max:.0f}°C)\n"
                     f"Cost: ${cost_ic:.2f}")
            
            fill_color = 'white'
            if name in path_nodes: fill_color = 'lightblue'
            elif name in supplier_nodes: fill_color = 'lightyellow'
            
            # 열 문제 시 빨간색
            color = 'blue'
            if (tj_max - tj) < 10: color = 'red'
            
            dot.node(name, label, color=color, fillcolor=fill_color, style=node_style)

    # 엣지
    for p_name, children in parent_to_children.items():
        for c_name in children:
            dot.edge(p_name, c_name)

    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"debug_solution_cost_{cost:.2f}_{timestamp}"
    output_path = os.path.join(result_dir, filename)
    
    try:
        dot.render(output_path, view=False, format='png', cleanup=True)
        print(f"✅ 상세 다이어그램 저장 완료: '{output_path}.png'")
    except Exception as e:
        print(f"❌ Graphviz render failed: {e}")
        
def run_interactive_debugger(config_file: str, n_max: int):
    """
    대화형으로 V7 환경(PocatEnv)을 한 스텝씩 실행하며
    Parameterized Action 마스킹 로직을 디버깅합니다.
    """
    
    # 1. V7 환경 초기화 (N_max 주입)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PocatEnv(
        generator_params={"config_file_path": config_file},
        device=device,
        N_max=n_max
    )
    td = env.reset(batch_size=1)
    
    static_node_names = env.generator.config.node_names
    num_nodes = env.N_max
    node_name_to_idx = {name: i for i, name in enumerate(static_node_names)}

    # Debug용으로 동적으로 스폰된 IC 이름을 추적하기 위한 버퍼.
    dynamic_node_names: List[str] = list(static_node_names)
    if len(dynamic_node_names) < num_nodes:
        dynamic_node_names.extend([None] * (num_nodes - len(dynamic_node_names)))
    spawn_name_counter: Dict[str, int] = {}


    print("="*60)
    print(f"🚀 V7 POCAT Interactive Debugger (N_MAX={n_max}) 🚀")
    print(f"Config: {config_file}")
    print("액션은 '이름'(예: LOAD_A) 또는 '인덱스'(예: 1)로 입력하세요.")
    print("'exit' 입력 시 종료, 'cost' 입력 시 현재 비용 확인.")
    print("="*60)

    step = 0
    while not td["done"].all():
        step += 1
        current_head_idx = td["trajectory_head"].item()
        current_head_name = get_node_name(current_head_idx, dynamic_node_names)
        
        print(f"\n--- Step {step} (Head: {current_head_name} [idx:{current_head_idx}]) ---")
        
        # 2. [V7] 3종 마스크 및 디버그 정보 가져오기
        #    (solver_env.py의 get_action_mask가 debug=True를 지원한다고 가정)

        mask_info = env.get_action_mask(td, debug=True)
        masks = {k: v[0] for k, v in mask_info.items() if "mask_" in k} # (B=1 제거)
        reasons = {k: v for k, v in mask_info.get("reasons", {}).items()}        # 3. [V7] Action Type 마스크 출력

        mask_type = masks["mask_type"] # (2,)
        can_connect = mask_type[0].item()
        can_spawn = mask_type[1].item()
        
        print(f"Action Type Mask: [Connect: {can_connect}, Spawn: {can_spawn}]")
        
        if not can_connect and not can_spawn:
            print("❌ STUCK: 가능한 액션 타입이 없습니다. (종료)")
            break
            
        # 4. 사용자로부터 Action Type 입력받기
        action_type = -1
        while action_type == -1:
            user_input = input("Select Action Type (0=Connect, 1=Spawn, exit): ").strip().lower()
            if user_input == 'exit': return
            
            if user_input == '0' and can_connect:
                action_type = 0
            elif user_input == '1' and can_spawn:
                action_type = 1
            else:
                print(f"  -> 잘못된 입력이거나 마스킹된 액션입니다.")

        # --- 5. 선택된 타입에 따라 세부 액션 처리 ---
        action_connect_idx = -1
        action_spawn_idx = -1
        
        if action_type == 0:
            # --- Connect ---
            print("\n  --- (Mode: Connect) ---")
            mask_connect = masks["mask_connect"] # (N_max,)
            valid_indices = torch.where(mask_connect)[0]

            # (디버그 정보 출력)
            print("  --- Reasons (Connect) ---")
            print(f"  base_valid_parents (저비용): {torch.where(reasons.get('base_valid_parents', torch.tensor([])))[0].tolist()}")
            print(f"  thermal_current_ok (고비용): {torch.where(reasons.get('thermal_current_ok', torch.tensor([])))[0].tolist()}")
            print(f"  is_active (상태): {torch.where(td['is_active_mask'][0])[0].tolist()}")
            print("  ---------------------------")

            print(f"  Valid Connect Targets ({len(valid_indices)}):")
            valid_actions_map = {}
            for idx in valid_indices:
                name = get_node_name(idx.item(), dynamic_node_names)
                print(f"    - {name} (idx: {idx.item()})")
                valid_actions_map[name.lower()] = idx.item()
                valid_actions_map[str(idx.item())] = idx.item()

            while action_connect_idx == -1:
                user_input = input("    Select Connect Target: ").strip()
                if user_input == 'exit': return
                key = user_input.lower()
                if key in valid_actions_map:
                    action_connect_idx = valid_actions_map[key]
                else:
                    print("    -> 잘못된 타겟입니다.")
            
            action_spawn_idx = 0 # (Spawn이 아니므로 0번 템플릿으로 더미 패딩)

        else:
            # --- Spawn ---
            print("\n  --- (Mode: Spawn) ---")
            mask_spawn = masks["mask_spawn"] # (N_max,)
            valid_indices = torch.where(mask_spawn)[0]

            # (디버그 정보 출력)
            print("  --- Reasons (Spawn) ---")
            print(f"  base_valid_parents (저비용): {torch.where(reasons.get('base_valid_parents', torch.tensor([])))[0].tolist()}")
            print(f"  thermal_current_ok (고비용): {torch.where(reasons.get('thermal_current_ok', torch.tensor([])))[0].tolist()}")
            print(f"  is_template (상태): {torch.where(td['is_template_mask'][0])[0].tolist()}")
            print("  ---------------------------")
            
            print(f"  Valid Spawn Templates ({len(valid_indices)}):")
            valid_actions_map = {}
            for idx in valid_indices:
                name = get_node_name(idx.item(), dynamic_node_names)
                print(f"    - {name} (idx: {idx.item()})")
                valid_actions_map[name.lower()] = idx.item()
                valid_actions_map[str(idx.item())] = idx.item()
                
            while action_spawn_idx == -1:
                user_input = input("    Select Spawn Template: ").strip()
                if user_input == 'exit': return
                key = user_input.lower()
                if key in valid_actions_map:
                    action_spawn_idx = valid_actions_map[key]
                else:
                    print("    -> 잘못된 템플릿입니다.")

            action_connect_idx = 0 # (Connect가 아니므로 0번 노드(BATT)로 더미 패딩)

        # 6. 환경 스텝 실행
        action_dict = {
            "action_type": torch.tensor([[action_type]], device=device),
            "connect_target": torch.tensor([[action_connect_idx]], device=device),
            "spawn_template": torch.tensor([[action_spawn_idx]], device=device),
        }
        
        if action_type == 1:
            slot_idx = td["next_empty_slot_idx"].item()
            template_idx = action_spawn_idx
            if 0 <= template_idx < len(static_node_names):
                base_name = static_node_names[template_idx]
            else:
                base_name = get_node_name(template_idx, dynamic_node_names)
            spawn_name_counter[base_name] = spawn_name_counter.get(base_name, 0) + 1
            display_name = f"{base_name}#{spawn_name_counter[base_name]}"
            if 0 <= slot_idx < len(dynamic_node_names):
                dynamic_node_names[slot_idx] = display_name

        td.set("action", action_dict)
        output = env.step(td)
        td = output["next"]

    print("\n🎉 Power Tree construction finished!")
    final_reward = output['reward'].item()
    print(f"Final Reward: {final_reward:.4f}")
    final_cost = td['current_cost'].item() # [수정] cost 변수 저장
    print(f"Final Cost (Staging+Current): ${td['current_cost'].item():.4f}")
    final_sleep_current = 0.0

    # [추가] 최종 암전류 계산 및 출력
    try:
        final_sleep_current = env._calculate_total_sleep_current(td).item()
        print(f"Final Sleep Current: {final_sleep_current * 1_000_000:.2f} µA")
        
        # 목표치 비교
        target_sleep = env.generator.config.constraints.get("max_sleep_current", 0.0)
        if target_sleep > 0:
            is_pass = final_sleep_current <= target_sleep
            status = "✅ PASS" if is_pass else f"❌ FAIL (Over {(final_sleep_current - target_sleep) * 1_000_000:.2f} µA)"
            print(f"Target Sleep Current: {target_sleep * 1_000_000:.2f} µA [{status}]")
            # [추가] 암전류 리워드(페널티) 계산
            violation = max(0, final_sleep_current - target_sleep)
            sleep_penalty_score = violation * PENALTY_WEIGHT_SLEEP
            print(f"Sleep Penalty Score: -{sleep_penalty_score:.4f}")

    except Exception as e:
        print(f"⚠️ Failed to calculate sleep current: {e}")

    visualize_debug_result(env, td, final_cost, final_sleep_current)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Debugger for V7 POCAT Env")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (.json) to debug.")
    # (config.yaml에서 N_MAX를 읽어올 수 없으므로, 명령줄 인자로 받음)
    parser.add_argument("--n_max", type=int, default=500, help="N_MAX (static max size) used by the model.")
    
    args = parser.parse_args()
    
    run_interactive_debugger(args.config_file, args.n_max)