# transformer_solver/debug_env.py

import torch
import argparse
import sys
import os
import pprint # (딕셔너리 출력을 위해)
from typing import Dict, List

# (common을 참조하므로, 프로젝트 루트 경로를 sys.path에 추가)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_solver.solver_env import PocatEnv, BATTERY_NODE_IDX
from transformer_solver.definitions import (
    FEATURE_INDEX, NODE_TYPE_LOAD, NODE_TYPE_IC, NODE_TYPE_EMPTY
)

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
    print(f"Final Cost (Staging+Current): ${td['current_cost'].item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Debugger for V7 POCAT Env")
    parser.add_argument("config_file", type=str, help="Path to the configuration file (.json) to debug.")
    # (config.yaml에서 N_MAX를 읽어올 수 없으므로, 명령줄 인자로 받음)
    parser.add_argument("--n_max", type=int, default=500, help="N_MAX (static max size) used by the model.")
    
    args = parser.parse_args()
    
    run_interactive_debugger(args.config_file, args.n_max)