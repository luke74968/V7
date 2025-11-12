# common/config_loader.py

import json
from typing import List, Dict, Tuple, Any

# data_classes에서 클래스들을 임포트합니다.
from .data_classes import Battery, Load, PowerIC, LDO, BuckConverter

def load_configuration_from_json(config_string: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    """
    JSON 설정 문자열을 파싱하여 데이터 객체들과 제약조건을 반환합니다.
    
    Args:
        config_string (str): JSON 파일의 내용을 담고 있는 문자열
        
    Returns:
        Tuple: (battery 객체, IC 객체 리스트, Load 객체 리스트, constraints 딕셔너리)
    """
    config = json.loads(config_string)
    
    # 1. 배터리 로드
    battery = Battery(**config['battery'])
    
    # 2. Power IC 로드
    available_ics = []
    for ic_data in config['available_ics']:
        
        # JSON의 'i_limit' (원본 스펙) 키를 
        # dataclass의 'original_i_limit' 필드로 매핑합니다.
        if 'i_limit' in ic_data:
            ic_data['original_i_limit'] = ic_data.pop('i_limit')
        # ---------------------
            
        ic_type = ic_data.pop('type')
        if ic_type == 'LDO':
            available_ics.append(LDO(**ic_data))
        elif ic_type == 'Buck':
            available_ics.append(BuckConverter(**ic_data))
        
        # 다른 코드(예: env_generator)가 원본 type을 참조할 수 있도록 
        # 딕셔너리에 'type' 키를 다시 추가해줍니다.
        ic_data['type'] = ic_type

    # 3. Load 로드
    loads = [Load(**load_data) for load_data in config['loads']]
    
    # 4. 제약조건 로드
    constraints = config['constraints']
    
    return battery, available_ics, loads, constraints

def load_configuration_from_file(filepath: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    """
    JSON 파일 경로를 입력받아 설정 객체들을 로드합니다.
    (V6의와 동일한 헬퍼 함수)
    
    Args:
        filepath (str): 로드할 config.json 파일의 경로
        
    Returns:
        Tuple: (battery 객체, IC 객체 리스트, Load 객체 리스트, constraints 딕셔너리)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json_config_string = f.read()
        print(f"✅ 설정 파일 로드 성공: '{filepath}'")
        return load_configuration_from_json(json_config_string)
    except FileNotFoundError:
        print(f"❌ 설정 파일 로드 실패: '{filepath}'을(를) 찾을 수 없습니다.")
        # 빈 리스트와 딕셔너리를 반환하여 프로그램이 즉시 중단되는 것을 방지
        return Battery(name="Error", voltage_min=0, voltage_max=0, capacity_mah=0), [], [], {}
    except Exception as e:
        print(f"❌ 설정 파일 처리 중 오류 발생: {e}")
        return Battery(name="Error", voltage_min=0, voltage_max=0, capacity_mah=0), [], [], {}