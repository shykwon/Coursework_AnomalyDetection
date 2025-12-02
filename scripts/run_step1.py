#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: 전처리 + 모델 학습 + Score 저장

Usage:
    python scripts/run_step1.py --dataset PSM
    python scripts/run_step1.py --dataset PSM --models DLinear
    python scripts/run_step1.py --dataset PSM --preprocess P_MM P_STD
    python scripts/run_step1.py --dataset PSM --gpu 0 --log_dir logs/
"""

import argparse
import sys
import os
import time
import logging
from datetime import datetime

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def setup_logging(log_dir: str, dataset: str, models: list, preprocess: list):
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_str = '-'.join(models)
    preprocess_str = '-'.join(preprocess)
    log_filename = f"{timestamp}_{dataset}_{models_str}_{preprocess_str}_step1.log"
    log_path = os.path.join(log_dir, log_filename)

    # 로거 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_path


def log(msg):
    """로깅 또는 print"""
    if logging.getLogger().handlers:
        logging.info(msg)
    else:
        print(msg)


import numpy as np

from data import DataLoader
from preprocessing import Scaler, EWMASmoother, MovingAverageDetrender
from models import DLinearModel, OmniAnomaly
from utils import ExperimentTracker

# 설정 파일
from config import get_model_config, DLINEAR_CONFIG, OMNIANOMALY_CONFIG


# ============================================================
# 전처리 파이프라인
# ============================================================

def apply_preprocessing(train_data, test_data, preprocess_id):
    """
    전처리 적용

    Args:
        train_data: 학습 데이터 (numpy array)
        test_data: 테스트 데이터 (numpy array)
        preprocess_id: 전처리 ID ('P_MM', 'P_STD', 'P_SM', 'P_DT')

    Returns:
        train_processed, test_processed
    """

    if preprocess_id == 'P_MM':
        # MinMax 정규화만
        scaler = Scaler(method='minmax')
        scaler.fit(train_data)
        return scaler.transform(train_data), scaler.transform(test_data)

    elif preprocess_id == 'P_STD':
        # Standard 정규화만
        scaler = Scaler(method='standard')
        scaler.fit(train_data)
        return scaler.transform(train_data), scaler.transform(test_data)

    elif preprocess_id == 'P_SM':
        # MinMax + EWMA Smoothing
        scaler = Scaler(method='minmax')
        scaler.fit(train_data)
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        smoother = EWMASmoother(span=5)
        smoother.fit(train_scaled)  # fit 추가
        return smoother.transform(train_scaled), smoother.transform(test_scaled)

    elif preprocess_id == 'P_DT':
        # MinMax + Detrending
        scaler = Scaler(method='minmax')
        scaler.fit(train_data)
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)

        detrender = MovingAverageDetrender(window=10)
        detrender.fit(train_scaled)
        return detrender.transform(train_scaled), detrender.transform(test_scaled)

    else:
        raise ValueError(f"Unknown preprocess_id: {preprocess_id}")


# ============================================================
# 모델 학습
# ============================================================

def train_and_score(model_name, train_data, test_data, config=None):
    """
    모델 학습 및 Anomaly Score 계산

    Args:
        model_name: 'DLinear' 또는 'OmniAnomaly'
        train_data: 전처리된 학습 데이터
        test_data: 전처리된 테스트 데이터
        config: 모델 설정 (optional)

    Returns:
        scores: Anomaly Score (테스트 데이터 길이)
        training_time: 학습 소요 시간 (초)
    """

    if model_name == 'DLinear':
        model = DLinearModel(
            seq_len=config.get('seq_len', 100) if config else 100,
            pred_len=config.get('pred_len', 1) if config else 1,
            epochs=config.get('epochs', 50) if config else 50,
            batch_size=config.get('batch_size', 32) if config else 32,
            lr=config.get('lr', 0.005) if config else 0.005,
            early_stopping=config.get('early_stopping', True) if config else True,
            patience=config.get('patience', 5) if config else 5,
            val_ratio=config.get('val_ratio', 0.1) if config else 0.1,
        )

    elif model_name == 'OmniAnomaly':
        # OmniAnomaly는 config dict를 받음 (논문 원본 설정)
        omni_config = {
            'window_length': config.get('window_size', 100) if config else 100,
            'hidden_dim': config.get('hidden_size', 500) if config else 500,
            'z_dim': config.get('z_dim', 3) if config else 3,
            'n_flows': config.get('n_flows', 20) if config else 20,  # 논문: 20
            'epochs': config.get('epochs', 20) if config else 20,
            'batch_size': config.get('batch_size', 50) if config else 50,
            'learning_rate': config.get('lr', 0.001) if config else 0.001,
            'weight_decay': config.get('weight_decay', 1e-4) if config else 1e-4,  # L2 정규화
            'early_stopping': config.get('early_stopping', True) if config else True,
            'patience': config.get('patience', 5) if config else 5,
            'val_ratio': config.get('val_ratio', 0.3) if config else 0.3,  # 논문: 30%
        }
        model = OmniAnomaly(omni_config)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # 학습
    start_time = time.time()
    model.fit(train_data, verbose=True)
    training_time = time.time() - start_time

    # Score 계산
    scores = model.get_anomaly_score(test_data)

    return scores, training_time


# ============================================================
# 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Step 1: Training + Score 저장')
    parser.add_argument('--dataset', type=str, default='PSM',
                        choices=['PSM', 'SWaT'], help='데이터셋')
    parser.add_argument('--models', nargs='+', default=['DLinear', 'OmniAnomaly'],
                        help='학습할 모델 목록')
    parser.add_argument('--preprocess', nargs='+',
                        default=['P_MM', 'P_STD', 'P_SM', 'P_DT'],
                        help='전처리 ID 목록')
    parser.add_argument('--epochs', type=int, default=None, help='학습 에폭 (미지정시 레퍼런스 기본값)')
    parser.add_argument('--batch_size', type=int, default=None, help='배치 크기 (미지정시 레퍼런스 기본값)')
    parser.add_argument('--seq_len', type=int, default=None, help='시퀀스 길이 (미지정시 레퍼런스 기본값)')
    parser.add_argument('--gpu', type=int, default=None, help='사용할 GPU ID (미지정시 자동)')
    parser.add_argument('--log_dir', type=str, default=None, help='로그 저장 디렉토리')

    args = parser.parse_args()

    # GPU 설정
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 로깅 설정
    if args.log_dir:
        log_path = setup_logging(args.log_dir, args.dataset, args.models, args.preprocess)
        logging.info(f"Log file: {log_path}")

    # ============================================================
    # 레퍼런스 기반 기본 설정
    # ============================================================
    # OmniAnomaly 원본: window=100, batch=50, lr=0.001, epochs=10
    # DLinear 원본: seq_len=336, batch=32, lr=0.005, epochs=10
    #
    # 데이터셋 특성 고려:
    # - PSM: 132K samples, 25 features → 중소규모
    # - SWaT: 395K samples, 51 features → 대규모
    # ============================================================

    REFERENCE_CONFIGS = {
        'DLinear': {
            'seq_len': 100,      # 이상탐지용으로 336→100 조정 (원본은 장기예측용)
            'pred_len': 1,       # 1-step ahead 예측
            'epochs': 10,        # 레퍼런스 기본값
            'batch_size': 32,    # 레퍼런스 기본값
            'lr': 0.005          # 레퍼런스 기본값
        },
        'OmniAnomaly': {
            # 논문 원본 설정 (KDD 2019)
            'window_size': 100,   # T+1 = 100
            'hidden_size': 500,   # GRU units
            'z_dim': 3,           # Latent dimension
            'n_flows': 20,        # Planar Normalizing Flow layers
            'epochs': 20,         # with early stopping
            'batch_size': 50,     # 논문 원본
            'lr': 0.001,          # Initial learning rate
            'val_ratio': 0.3,     # 30% validation split
            'weight_decay': 1e-4  # L2 regularization
        }
    }

    log("=" * 70)
    log("Step 1: 전처리 + 모델 학습 + Score 저장")
    log("=" * 70)
    log(f"Dataset: {args.dataset}")
    log(f"Models: {args.models}")
    log(f"Preprocess: {args.preprocess}")
    if args.gpu is not None:
        log(f"GPU: {args.gpu}")
    log("=" * 70)

    # 데이터 로드
    data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', args.dataset)
    loader = DataLoader(args.dataset, data_path)
    train_data = loader.load_train()
    test_data, test_labels = loader.load_test()

    train_np = train_data.values
    test_np = test_data.values

    log(f"Train shape: {train_np.shape}")
    log(f"Test shape: {test_np.shape}")
    log(f"Anomaly ratio: {test_labels.mean()*100:.2f}%")

    # ExperimentTracker 초기화
    tracker = ExperimentTracker(base_dir=os.path.join(PROJECT_ROOT, 'outputs'))

    # 모델 설정 (CLI 인자로 오버라이드 가능)
    model_configs = {}

    for model_name in ['DLinear', 'OmniAnomaly']:
        config = REFERENCE_CONFIGS[model_name].copy()

        # CLI 오버라이드
        if args.epochs is not None:
            config['epochs'] = args.epochs
        if args.batch_size is not None:
            config['batch_size'] = args.batch_size
        if args.seq_len is not None:
            if model_name == 'DLinear':
                config['seq_len'] = args.seq_len
            else:
                config['window_size'] = args.seq_len

        model_configs[model_name] = config

    # 설정 출력
    log("📋 Model Configurations (Reference-based):")
    for name, cfg in model_configs.items():
        log(f"  {name}: {cfg}")

    # 실험 실행
    total_exp = len(args.preprocess) * len(args.models)
    current_exp = 0

    for preprocess_id in args.preprocess:
        log(f"{'─' * 50}")
        log(f"전처리: {preprocess_id}")
        log(f"{'─' * 50}")

        train_processed, test_processed = apply_preprocessing(
            train_np, test_np, preprocess_id
        )
        log(f"  전처리 완료: {train_processed.shape}")

        for model_name in args.models:
            current_exp += 1
            log(f"[{current_exp}/{total_exp}] {model_name} + {preprocess_id}")

            try:
                config = model_configs.get(model_name, {})
                scores, training_time = train_and_score(
                    model_name,
                    train_processed,
                    test_processed,
                    config
                )

                # 기록
                tracker.log_training(
                    model=model_name,
                    preprocess=preprocess_id,
                    dataset=args.dataset,
                    scores=scores,
                    training_time=training_time,
                    config=config
                )

            except Exception as e:
                log(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()

    # 결과 요약
    log("=" * 70)
    log("Step 1 완료!")
    log("=" * 70)
    tracker.print_summary()

    # 저장된 Score 확인
    available = tracker.list_available_scores()
    log(f"저장된 Score: {len(available)}개")
    for s in available:
        log(f"  - {s['model']}_{s['preprocess']}_{s['dataset']}")


if __name__ == '__main__':
    main()
