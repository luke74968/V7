# or_tools_solver/main.py
# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
"""
OR-Tools 솔버 실행 (or_tools_solver/main.py)

이 파일은 OR-Tools 솔버를 실행하는 메인(Entry Point)입니다.

작업 순서:
1. `common.config_loader`를 
   사용하여 설정 파일(.json)을 로드합니다.
2. `common.ic_preprocessor`의 `expand_ic_instances` (OR-Tools용)를 
   호출하여 모든 IC 복제본 인스턴스와 그룹 정보를 생성합니다.
3. `common.ic_preprocessor`의 `prune_dominated_ics`를 
   호출하여 불필요한 IC 인스턴스를 제거(Pruning)합니다.
4. `or_tools_solver.core`의 `create_solver_model`을 
   호출하여 CP-SAT 모델을 생성합니다.
5. CP-SAT 솔버를 실행하여 '대표해'를 탐색합니다.
6. `or_tools_solver.core`의 `find_all_load_distributions`를 
   호출하여 '병렬해'를 탐색합니다.
7. `or_tools_solver.solution_visualizer`를 
   사용하여 유효한 해를 검증하고 시각화(PNG)합니다.
"""
import sys
import argparse
import os 
from ortools.sat.python import cp_model

# --- 공용(common) 패키지 임포트 ---
from common.config_loader import load_configuration_from_file
from common.ic_preprocessor import expand_ic_instances, prune_dominated_ics

# --- OR-Tools 솔버 패키지 임포트 ---
from or_tools_solver.core import (
    create_solver_model, find_all_load_distributions 
    # SolutionLogger는 main에서 사용하지 않으므로 제거
)
from or_tools_solver.solution_visualizer import (
    check_solution_validity, print_and_visualize_one_solution
)


def main():
    """메인 실행 함수 """
    
    parser = argparse.ArgumentParser(description="POCAT OR-Tools Solver")
    parser.add_argument("config_filename", type=str, help="Path to the configuration file (.json)")
    parser.add_argument("--max_sleep_current", type=float, default=None, help="Override the max_sleep_current constraint (in Amperes).")
    args = parser.parse_args()
    
    print(f"📖 설정 파일 '{args.config_filename}' 로딩...")

    # Config Loader 사용
    battery, available_ics, loads, constraints = load_configuration_from_file(args.config_filename)
    if not battery or not loads:
        print("❌ 설정 파일 로드에 실패하여 종료합니다.")
        return

    if args.max_sleep_current is not None:
        original_value = constraints.get('max_sleep_current', 'N/A')
        print(f"⚡ 암전류 제약조건 변경: {original_value} -> {args.max_sleep_current} A")
        constraints['max_sleep_current'] = args.max_sleep_current

    # Preprocessor 호출
    candidate_ics, ic_groups = expand_ic_instances(available_ics, loads, battery, constraints)
    
    # Pruning 호출
    pruned_candidate_ics = prune_dominated_ics(candidate_ics)

    original_count = len(candidate_ics)
    pruned_count = len(pruned_candidate_ics)
    print(f"   - {original_count - pruned_count}개의 지배되는 IC 인스턴스 제거 완료!")
    print(f"   - 남은 후보 IC 인스턴스: {pruned_count}개")
    
    # Pruning된 결과를 반영하여 ic_groups 정리 
    pruned_candidate_names = {ic.name for ic in pruned_candidate_ics}
    sanitized_ic_groups = {}
    for group_key, group_list in ic_groups.items():
        sanitized_group_list = [name for name in group_list if name in pruned_candidate_names]
        if len(sanitized_group_list) > 1:
            sanitized_ic_groups[group_key] = sanitized_group_list

    # Core를 호출하여 CP-SAT 모델 생성 
    model, edges, ic_is_used = create_solver_model(
        pruned_candidate_ics, loads, battery, constraints, sanitized_ic_groups
    )
    
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 300.0 
    
    print("\n🔍 최적의 대표 솔루션 탐색 시작...")
    status = solver.Solve(model)
    
    # 결과 처리 
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\n🎉 탐색 완료! (상태: {solver.StatusName(status)})")
        
        # SolutionLogger가 아닌,
        # solver 객체에서 직접 결과를 가져옵니다.
        base_solution = {
            "score": solver.ObjectiveValue(),
            "cost": solver.ObjectiveValue() / 10000.0,
            "used_ic_names": {name for name, var in ic_is_used.items() if solver.Value(var)},
            "active_edges": [(p, c) for (p, c), var in edges.items() if solver.Value(var)]
        }
        
        # 병렬해 탐색 
        find_all_load_distributions(
            base_solution, 
            pruned_candidate_ics, 
            loads, 
            battery, 
            constraints,
            viz_func=print_and_visualize_one_solution,
            check_func=check_solution_validity
        )
        
    else:
        print("\n❌ 유효한 솔루션을 찾지 못했습니다.")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    main()