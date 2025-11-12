# V7/common/data_classes.py

from dataclasses import dataclass, field
from typing import Dict, Optional

# --- 1. 배터리 (전원 공급원) ---

@dataclass
class Battery:
    """
    전원 공급원인 배터리의 사양을 정의합니다.
    """
    name: str
    voltage_min: float
    voltage_max: float
    capacity_mah: int
    vout: float = 0.0 # 평균 전압 (나중에 계산됨)

# --- 2. 부하 (전력 소비자) ---

@dataclass
class Load:
    """
    전력을 소비하는 부하(Load)의 요구사항을 정의합니다.
    """
    name: str
    voltage_req_min: float
    voltage_req_max: float
    voltage_typical: float
    current_active: float
    current_sleep: float
    independent_rail_type: Optional[str] = None
    always_on_in_sleep: bool = False

# --- 3. Power IC (전력 변환기 - 부모 클래스) ---

@dataclass
class PowerIC:
    """
    전력 변환 IC (LDO, Buck 등)의 공통 사양을 정의하는 기본 클래스입니다.
    """
    name: str
    vin_min: float
    vin_max: float
    vout_min: float
    vout_max: float
    original_i_limit: float = 0.0 # JSON('i_limit')에서 직접 로드되는 '원본 스펙' 값입니다.
    operating_current: float  # IC 자체의 동작 전류 (Iop)
    quiescent_current: float  # IC 자체의 대기 전류 (Iq)
    cost: float
    theta_ja: float
    t_junction_max: int
    
    shutdown_current: Optional[float] = None # 차단(Shutdown) 모드 전류
    
    # '특화된 인스턴스' 생성 시 채워질 필드들
    vin: float = 0.0
    vout: float = 0.0
    i_limit: float = 0.0 # '인스턴스 확장' 시 계산되어 채워질 '유효 한계값'입니다.

    # --- 1. 활성(Active) 모드 계산 메소드 ---
    
    def calculate_power_loss(self, vin: float, i_out: float) -> float:
        raise NotImplementedError

    def calculate_active_input_current(self, vin: float, i_out: float) -> float:
        raise NotImplementedError

    # --- 2. 절전(Sleep) 모드 계산 메소드  ---

    def get_self_sleep_consumption(self, is_on_ao_path: bool, parent_is_on_ao_path: bool) -> float:
        """
        IC '자체'의 절전 소모 전류를 반환합니다.
        (3-state 로직 캡슐화)
        """
        if is_on_ao_path:
            # 상태 1: "Always-On" 경로 -> Iop 소모
            return self.operating_current
        
        elif parent_is_on_ao_path:
            # 상태 2: "비-AO"지만 부모가 켜짐 출력 전류 X -> I_shut 또는 Iq 소모
            if self.shutdown_current is not None and self.shutdown_current > 0:
                return self.shutdown_current
            return self.quiescent_current
        
        else:
            # 상태 3: "완전 차단" -> 0 소모
            return 0.0

    def calculate_sleep_input_for_children(self, vin: float, i_out_sleep: float) -> float:
        """
        절전 상태에서 '자식'들에게 i_out_sleep을 공급하기 위해 
        필요한 입력 전류(A)를 계산합니다. (IC 자체 소모 전류는 제외)
        """
        raise NotImplementedError

# --- 4. LDO (PowerIC의 자식 클래스) ---

@dataclass
class LDO(PowerIC):
    type: str = "LDO"
    v_dropout: float = 0.0

    def calculate_power_loss(self, vin: float, i_out: float) -> float:
        # 손실 = (Dropout 손실) + (IC 자체 동작 손실)
        return ((vin - self.vout) * i_out) + (vin * self.operating_current)

    def calculate_active_input_current(self, vin: float, i_out: float) -> float:
        # I_in = I_out + I_op
        return i_out + self.operating_current

    def calculate_sleep_input_for_children(self, vin: float, i_out_sleep: float) -> float:
        # LDO는 I_in = I_out (자체 소모는 별도 계산됨)
        return i_out_sleep

# --- 5. Buck (PowerIC의 자식 클래스)  ---

@dataclass
class BuckConverter(PowerIC):
    """
    Buck Converter (DCDC)의 특성을 정의합니다.
    활성 90%, 절전 35%의 고정 효율을 사용합니다.
    """
    type: str = "Buck"
   

    def calculate_power_loss(self, vin: float, i_out: float) -> float:
        # 손실 = (변환 손실) + (IC 자체 동작 손실)
        p_out = self.vout * i_out
        
        # 활성 모드 효율 90% (0.9) 고정
        eff = 0.9 
        if eff == 0: return float('inf')
        
        conversion_loss = (p_out / eff) - p_out
        return conversion_loss + (vin * self.operating_current) 

    def calculate_active_input_current(self, vin: float, i_out: float) -> float:
        # I_in = (P_in / V_in) + I_op
        if vin == 0: return float('inf')
        p_out = self.vout * i_out
        
        eff = 0.9
        if eff == 0: return float('inf')
        
        p_in = p_out / eff
        return (p_in / vin) + self.operating_current

    def calculate_sleep_input_for_children(self, vin: float, i_out_sleep: float) -> float:
        """
        Buck의 절전 상태 입력 전류를 계산합니다.
        절전 효율 35% (0.35) 고정
        """
        if vin == 0: return float('inf')
        if i_out_sleep == 0: return 0.0

        eff_sleep = 0.35 # 고정 효율 35%
        
        p_out_sleep = self.vout * i_out_sleep
        p_in_sleep = p_out_sleep / eff_sleep
        
        return p_in_sleep / vin