# -*- coding: utf-8 -*-
"""
Evaluation base module
Abstract base class for all metrics
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class BaseMetric(ABC):
    """
    Metric abstract base class

    All metrics inherit from this class.

    Metrics:
    - Precision: TP / (TP + FP) - How many predicted anomalies are actual anomalies
    - Recall: TP / (TP + FN) - How many actual anomalies are detected
    - F1: 2 * (Precision * Recall) / (Precision + Recall)
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute metric

        Args:
            y_true: Ground truth labels (0: normal, 1: anomaly)
            y_pred: Predicted labels (0: normal, 1: anomaly)

        Returns:
            Dict: Metric results (precision, recall, f1, etc.)
        """
        pass

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Validate input arrays"""
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            )

        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1")

        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError("y_pred must contain only 0 and 1")

    def _compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute confusion matrix components

        Returns:
            Dict: TP, TN, FP, FN counts
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)}

    def _compute_precision_recall_f1(self, tp: int, fp: int, fn: int) -> Dict:
        """
        Compute precision, recall, F1 from confusion matrix

        Returns:
            Dict: precision, recall, f1
        """
        # ============================================================
        # TODO(human): Precision, Recall, F1 계산
        # ============================================================
        # Precision = TP / (TP + FP)  - 예측한 것 중 맞은 비율
        # Recall = TP / (TP + FN)     - 실제 이상치 중 탐지한 비율
        # F1 = 2 * P * R / (P + R)    - 조화 평균
        #
        # 주의: 0으로 나누기 방지 필요

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        
        if precision is None:
            raise NotImplementedError(
                "TODO(human): _compute_precision_recall_f1()를 구현해주세요!"
            )

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
