#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: 후처리 + 평가 (Score 로드, 재학습 없음)

Usage:
    python scripts/run_step2.py --dataset PSM
    python scripts/run_step2.py --dataset PSM --postprocess T1 T2 T3
    python scripts/run_step2.py --dataset PSM --models DLinear
    python scripts/run_step2.py --dataset PSM --log_dir logs/

후처리 전략:
    T1: Fixed Threshold (3σ)
    T2: Fixed Threshold (99th Percentile)
    T3: EWMA Adaptive Threshold
    T4: Score Smoothing + Fixed (3σ)
    T5: Score Smoothing + EWMA Adaptive
"""

import argparse
import sys
import os
import logging
from datetime import datetime

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


def setup_logging(log_dir: str, dataset: str, postprocess: list):
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    postprocess_str = '-'.join(postprocess)
    log_filename = f"{timestamp}_{dataset}_{postprocess_str}_step2.log"
    log_path = os.path.join(log_dir, log_filename)

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
import pandas as pd

from data import DataLoader
from preprocessing import EWMASmoother
from postprocessing import FixedThreshold, EWMAThreshold
from evaluation import PointF1, PointAdjustF1, AUCMetrics
from utils import ExperimentTracker


# ============================================================
# 후처리 전략 정의
# ============================================================

def clip_extreme_scores(scores, lower_percentile=0.1, upper_percentile=99.9):
    """
    극단값 클리핑 (수치 안정성)

    Args:
        scores: Anomaly Score 배열
        lower_percentile: 하위 백분위
        upper_percentile: 상위 백분위

    Returns:
        클리핑된 scores
    """
    lower = np.percentile(scores, lower_percentile)
    upper = np.percentile(scores, upper_percentile)
    return np.clip(scores, lower, upper)


def apply_postprocess(scores, postprocess_id, train_scores=None):
    """
    후처리 전략 적용

    Args:
        scores: Anomaly Score 배열
        postprocess_id: 후처리 ID ('T1' ~ 'T5')
        train_scores: 학습 데이터 Score (임계값 학습용, optional)

    Returns:
        predictions: 이진 예측 (0/1)
        threshold_value: 사용된 임계값
    """
    # 극단값 클리핑 (수치 안정성)
    scores = clip_extreme_scores(scores)

    fit_scores = train_scores if train_scores is not None else scores
    if train_scores is not None:
        fit_scores = clip_extreme_scores(fit_scores)

    if postprocess_id == 'T1':
        # Fixed Threshold (3σ)
        threshold = FixedThreshold(method='sigma', n_sigma=3.0)
        threshold.fit(fit_scores)
        predictions = threshold.apply(scores)
        return predictions, threshold.threshold_

    elif postprocess_id == 'T2':
        # Fixed Threshold (99th Percentile)
        threshold = FixedThreshold(method='percentile', percentile=99.0)
        threshold.fit(fit_scores)
        predictions = threshold.apply(scores)
        return predictions, threshold.threshold_

    elif postprocess_id == 'T3':
        # EWMA Adaptive Threshold
        threshold = EWMAThreshold(span=100, n_sigma=3.0)
        threshold.fit(fit_scores)
        predictions = threshold.apply(scores)
        # EWMA는 동적 임계값이므로 평균값 반환
        return predictions, np.mean(threshold.threshold_)

    elif postprocess_id == 'T4':
        # Score Smoothing + Fixed (3σ)
        smoother = EWMASmoother(span=5)
        smoother.fit(scores.reshape(-1, 1))  # fit 추가
        smoothed_scores = smoother.transform(scores.reshape(-1, 1)).flatten()

        threshold = FixedThreshold(method='sigma', n_sigma=3.0)
        if train_scores is not None:
            smoothed_train = smoother.transform(train_scores.reshape(-1, 1)).flatten()
            threshold.fit(smoothed_train)
        else:
            threshold.fit(smoothed_scores)

        predictions = threshold.apply(smoothed_scores)
        return predictions, threshold.threshold_

    elif postprocess_id == 'T5':
        # Score Smoothing + EWMA Adaptive
        smoother = EWMASmoother(span=5)
        smoother.fit(scores.reshape(-1, 1))  # fit 추가
        smoothed_scores = smoother.transform(scores.reshape(-1, 1)).flatten()

        threshold = EWMAThreshold(span=100, n_sigma=3.0)
        if train_scores is not None:
            smoothed_train = smoother.transform(train_scores.reshape(-1, 1)).flatten()
            threshold.fit(smoothed_train)
        else:
            threshold.fit(smoothed_scores)

        predictions = threshold.apply(smoothed_scores)
        return predictions, np.mean(threshold.threshold_)

    else:
        raise ValueError(f"Unknown postprocess_id: {postprocess_id}")


# ============================================================
# 평가 함수
# ============================================================

def evaluate(predictions, labels, scores=None):
    """
    평가 지표 계산

    Args:
        predictions: 이진 예측 (0/1)
        labels: Ground Truth (0/1)
        scores: 원본 이상치 점수 (AUC 계산용, optional)

    Returns:
        metrics: 평가 지표 딕셔너리
    """
    # Point F1
    point_metric = PointF1()
    point_result = point_metric.compute(labels, predictions)

    # PA F1
    pa_metric = PointAdjustF1()
    pa_result = pa_metric.compute(labels, predictions)

    metrics = {
        'point_precision': point_result['precision'],
        'point_recall': point_result['recall'],
        'point_f1': point_result['f1'],
        'pa_precision': pa_result['precision'],
        'pa_recall': pa_result['recall'],
        'pa_f1': pa_result['f1']
    }

    # AUC 지표 (scores가 있을 때만)
    if scores is not None:
        auc_metric = AUCMetrics()
        auc_result = auc_metric.compute(labels, scores)
        metrics['roc_auc'] = auc_result['roc_auc']
        metrics['pr_auc'] = auc_result['pr_auc']

    return metrics


# ============================================================
# 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Step 2: 후처리 + 평가')
    parser.add_argument('--dataset', type=str, default='PSM',
                        choices=['PSM', 'SWaT'], help='데이터셋')
    parser.add_argument('--models', nargs='+', default=None,
                        help='평가할 모델 (미지정시 전체)')
    parser.add_argument('--preprocess', nargs='+', default=None,
                        help='평가할 전처리 (미지정시 전체)')
    parser.add_argument('--postprocess', nargs='+',
                        default=['T1', 'T2', 'T3', 'T4', 'T5'],
                        help='후처리 전략 목록')
    parser.add_argument('--log_dir', type=str, default=None, help='로그 저장 디렉토리')

    args = parser.parse_args()

    # 로깅 설정
    if args.log_dir:
        log_path = setup_logging(args.log_dir, args.dataset, args.postprocess)
        logging.info(f"Log file: {log_path}")

    log("=" * 70)
    log("Step 2: 후처리 + 평가 (Score 로드, 재학습 없음)")
    log("=" * 70)
    log(f"Dataset: {args.dataset}")
    log(f"Postprocess: {args.postprocess}")
    log("=" * 70)

    # Ground Truth 로드
    data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', args.dataset)
    loader = DataLoader(args.dataset, data_path)
    _, test_labels = loader.load_test()

    log(f"Test labels shape: {test_labels.shape}")
    log(f"Anomaly ratio: {test_labels.mean()*100:.2f}%")

    # ExperimentTracker 초기화
    tracker = ExperimentTracker(base_dir=os.path.join(PROJECT_ROOT, 'outputs'))

    # 사용 가능한 Score 목록
    available_scores = tracker.list_available_scores()

    # 필터링
    if args.models:
        available_scores = [s for s in available_scores if s['model'] in args.models]
    if args.preprocess:
        available_scores = [s for s in available_scores if s['preprocess'] in args.preprocess]

    # 해당 데이터셋만
    available_scores = [s for s in available_scores if s['dataset'] == args.dataset]

    if not available_scores:
        log("\n⚠️ 사용 가능한 Score가 없습니다.")
        log("먼저 run_step1.py를 실행하세요.")
        return

    log(f"\n평가할 Score: {len(available_scores)}개")
    for s in available_scores:
        log(f"  - {s['model']}_{s['preprocess']}_{s['dataset']}")

    # 실험 실행
    total_exp = len(available_scores) * len(args.postprocess)
    current_exp = 0

    results = []

    for score_info in available_scores:
        model = score_info['model']
        preprocess = score_info['preprocess']

        log(f"\n{'─' * 50}")
        log(f"Score: {model} + {preprocess}")
        log(f"{'─' * 50}")

        # Score 로드
        scores = tracker.load_scores(model, preprocess, args.dataset)
        log(f"  Score 로드 완료: shape={scores.shape}")

        # 길이 맞추기
        min_len = min(len(scores), len(test_labels))
        scores = scores[:min_len]
        labels = test_labels[:min_len]

        for postprocess_id in args.postprocess:
            current_exp += 1
            log(f"\n  [{current_exp}/{total_exp}] {postprocess_id}")

            try:
                # 후처리 적용
                predictions, threshold_value = apply_postprocess(scores, postprocess_id)

                # 평가
                metrics = evaluate(predictions, labels, scores=scores)

                # 기록
                tracker.log_evaluation(
                    model=model,
                    preprocess=preprocess,
                    postprocess=postprocess_id,
                    dataset=args.dataset,
                    metrics=metrics,
                    threshold_value=threshold_value
                )

                # 결과 저장
                results.append({
                    'model': model,
                    'preprocess': preprocess,
                    'postprocess': postprocess_id,
                    **metrics
                })

                log(f"    Point F1: {metrics['point_f1']:.4f}")
                log(f"    PA F1: {metrics['pa_f1']:.4f}")
                log(f"    ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in metrics else "")
                log(f"    PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if 'pr_auc' in metrics else "")

            except Exception as e:
                log(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()

    # 결과 요약
    log("\n" + "=" * 70)
    log("Step 2 완료!")
    log("=" * 70)

    if results:
        df = pd.DataFrame(results)

        # PA F1 기준 정렬
        df_sorted = df.sort_values('pa_f1', ascending=False)

        log("\n📊 PA F1 Top 5:")
        log("-" * 70)
        top5 = df_sorted.head(5)[['model', 'preprocess', 'postprocess', 'pa_f1', 'point_f1']]
        log(top5.to_string(index=False))

        # 피벗 테이블
        log("\n📊 후처리별 PA F1 (평균):")
        log("-" * 70)
        pivot = df.pivot_table(
            index='postprocess',
            columns='model',
            values='pa_f1',
            aggfunc='mean'
        ).round(4)
        log(pivot.to_string())

    tracker.print_summary()


if __name__ == '__main__':
    main()
