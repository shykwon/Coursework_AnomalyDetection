#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
실험 설정 파일

레퍼런스 기반 하이퍼파라미터 설정
- OmniAnomaly: https://github.com/NetManAIOps/OmniAnomaly
- DLinear: https://github.com/cure-lab/LTSF-Linear
"""

# ============================================================
# 레퍼런스 기반 기본 설정
# ============================================================

# DLinear 설정
# 원본 레퍼런스 (장기 예측용): seq_len=336, batch=32, lr=0.005, epochs=10
# 이상탐지 조정: seq_len=100 (윈도우 크기 축소)
DLINEAR_CONFIG = {
    'seq_len': 100,      # 원본 336 → 이상탐지용 100
    'pred_len': 1,       # 1-step ahead 예측 (이상탐지)
    'epochs': 50,        # Early Stopping 사용시 넉넉하게 설정
    'batch_size': 32,    # 레퍼런스 기본값
    'lr': 0.005,         # 레퍼런스 기본값
    'individual': False, # 변수별 독립 학습 여부
    # Early Stopping 설정
    'early_stopping': True,
    'patience': 5,       # 5 epoch 연속 개선 없으면 종료
    'val_ratio': 0.1     # 10% validation split
}

# OmniAnomaly 설정
# 원본 레퍼런스: window=100, hidden=500, z_dim=3, batch=50, lr=0.001, epochs=10
# 원본에 early_stop=True 옵션 있음
OMNIANOMALY_CONFIG = {
    'window_size': 100,  # 레퍼런스 기본값
    'hidden_size': 500,  # 레퍼런스 기본값 (GRU hidden)
    'z_dim': 3,          # 레퍼런스 기본값 (latent dim)
    'dense_dim': 500,    # 레퍼런스 기본값
    'nf_layers': 20,     # Normalizing Flow 레이어 수
    'epochs': 50,        # Early Stopping 사용시 넉넉하게 설정
    'batch_size': 50,    # 레퍼런스 기본값
    'lr': 0.001,         # 레퍼런스 기본값
    # Early Stopping 설정 (원본 레퍼런스에서도 사용)
    'early_stopping': True,
    'patience': 5,
    'val_ratio': 0.1
}

# ============================================================
# 데이터셋별 조정 설정 (선택적)
# ============================================================

# PSM 데이터셋: 132K train, 25 features
# - 중소규모 데이터 → 기본 설정으로 충분
PSM_ADJUSTMENTS = {
    'DLinear': {},
    'OmniAnomaly': {}
}

# SWaT 데이터셋: 395K train, 51 features
# - 대규모 데이터 → 배치 크기 증가 고려
SWAT_ADJUSTMENTS = {
    'DLinear': {
        'batch_size': 64  # 메모리 여유 시 증가 가능
    },
    'OmniAnomaly': {
        'batch_size': 64  # 메모리 여유 시 증가 가능
    }
}


# ============================================================
# 설정 가져오기 함수
# ============================================================

def get_model_config(model_name: str, dataset: str = 'PSM') -> dict:
    """
    모델 설정 반환

    Args:
        model_name: 'DLinear' 또는 'OmniAnomaly'
        dataset: 'PSM' 또는 'SWaT'

    Returns:
        설정 딕셔너리
    """
    if model_name == 'DLinear':
        config = DLINEAR_CONFIG.copy()
    elif model_name == 'OmniAnomaly':
        config = OMNIANOMALY_CONFIG.copy()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 데이터셋별 조정 적용
    if dataset == 'PSM':
        adjustments = PSM_ADJUSTMENTS.get(model_name, {})
    elif dataset == 'SWaT':
        adjustments = SWAT_ADJUSTMENTS.get(model_name, {})
    else:
        adjustments = {}

    config.update(adjustments)
    return config


def print_all_configs():
    """모든 설정 출력"""
    print("=" * 60)
    print("📋 Experiment Configurations (Reference-based)")
    print("=" * 60)

    print("\n[DLinear]")
    print(f"  Reference: LTSF-Linear (https://github.com/cure-lab/LTSF-Linear)")
    for k, v in DLINEAR_CONFIG.items():
        print(f"    {k}: {v}")

    print("\n[OmniAnomaly]")
    print(f"  Reference: OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)")
    for k, v in OMNIANOMALY_CONFIG.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    print_all_configs()
