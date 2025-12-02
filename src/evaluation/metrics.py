# -*- coding: utf-8 -*-
"""
Evaluation metrics module
Point-wise F1 and Point Adjustment F1 implementation
"""

from typing import Dict, List, Tuple

import numpy as np

from .base import BaseMetric


class PointF1(BaseMetric):
    """
    Point-wise F1 Score

    Each time point is evaluated independently.
    Standard precision, recall, F1 calculation.
    """

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute point-wise F1

        Args:
            y_true: Ground truth (0: normal, 1: anomaly)
            y_pred: Predictions (0: normal, 1: anomaly)

        Returns:
            Dict: precision, recall, f1, confusion matrix
        """
        self._validate_inputs(y_true, y_pred)

        # Confusion matrix
        cm = self._compute_confusion_matrix(y_true, y_pred)

        # Precision, Recall, F1
        metrics = self._compute_precision_recall_f1(cm['TP'], cm['FP'], cm['FN'])

        return {
            'metric_name': 'PointF1',
            **metrics,
            **cm
        }


class PointAdjustF1(BaseMetric):
    """
    Point Adjustment F1 Score (PA F1)

    If ANY point within an anomaly segment is detected,
    ALL points in that segment are considered as detected.

    This is commonly used in time series anomaly detection
    because detecting any point in an anomaly event is valuable.

    Reference:
    - Xu et al., "Unsupervised Anomaly Detection via Variational Auto-Encoder" (2018)
    """

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute Point Adjustment F1

        Args:
            y_true: Ground truth (0: normal, 1: anomaly)
            y_pred: Predictions (0: normal, 1: anomaly)

        Returns:
            Dict: precision, recall, f1 (with point adjustment)
        """
        self._validate_inputs(y_true, y_pred)

        # Apply point adjustment
        y_pred_adjusted = self._point_adjust(y_true, y_pred)

        # Confusion matrix with adjusted predictions
        cm = self._compute_confusion_matrix(y_true, y_pred_adjusted)

        # Precision, Recall, F1
        metrics = self._compute_precision_recall_f1(cm['TP'], cm['FP'], cm['FN'])

        return {
            'metric_name': 'PointAdjustF1',
            **metrics,
            **cm,
            'adjusted_predictions': y_pred_adjusted
        }

    def _point_adjust(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Apply point adjustment

        If any point in an anomaly segment is predicted as anomaly,
        all points in that segment are considered as correctly predicted.

        Args:
            y_true: Ground truth
            y_pred: Original predictions

        Returns:
            Adjusted predictions
        """
        # ============================================================
        # TODO(human): Point Adjustment 구현
        # ============================================================
        # 1. y_true에서 연속된 이상치 구간(segment) 찾기
        # 2. 각 구간에서 y_pred가 1인 점이 하나라도 있으면
        # 3. 해당 구간 전체를 1로 설정
        #
        # 예시:
        # y_true = [0,0,1,1,1,0,0,1,1,0]
        # y_pred = [0,0,0,1,0,0,0,0,1,0]
        # 결과  = [0,0,1,1,1,0,0,1,1,0]  (구간 내 하나라도 맞추면 전체 정답)

        # 이상치 구간 찾기
        segments = self._find_anomaly_segments(y_true)

        # 조정된 예측 생성
        y_pred_adjusted = y_pred.copy()

        for start, end in segments:
            segment_pred = y_pred[start:end]
            # TODO(human): 구간 내에 1이 하나라도 있으면 전체를 1로 설정
            if np.any(segment_pred == 1):  # TODO(human): 조건을 구현하세요
                y_pred_adjusted[start:end] = 1

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 📖 정답 (막히면 아래 주석을 해제하세요)
        # ────────────────────────────────────────────────────────────
        # for start, end in segments:
        #     segment_pred = y_pred[start:end]
        #     if np.any(segment_pred == 1):  # 구간 내에 1이 하나라도 있으면
        #         y_pred_adjusted[start:end] = 1  # 전체를 1로
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        return y_pred_adjusted

    def _find_anomaly_segments(self, y_true: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find continuous anomaly segments

        Args:
            y_true: Ground truth labels

        Returns:
            List of (start, end) tuples for each segment
        """
        segments = []
        in_segment = False
        start = 0

        for i, val in enumerate(y_true):
            if val == 1 and not in_segment:
                # Segment start
                start = i
                in_segment = True
            elif val == 0 and in_segment:
                # Segment end
                segments.append((start, i))
                in_segment = False

        # Handle case where segment extends to end
        if in_segment:
            segments.append((start, len(y_true)))

        return segments


class AUCMetrics(BaseMetric):
    """
    AUC-based Metrics (ROC-AUC, PR-AUC)

    These metrics use anomaly scores (continuous values) instead of binary predictions.
    More informative for threshold-independent evaluation.

    - ROC-AUC: Area Under ROC Curve (TPR vs FPR)
    - PR-AUC: Area Under Precision-Recall Curve (better for imbalanced data)
    """

    def compute(self, y_true: np.ndarray, scores: np.ndarray) -> Dict:
        """
        Compute AUC metrics

        Args:
            y_true: Ground truth (0: normal, 1: anomaly)
            scores: Anomaly scores (continuous, higher = more anomalous)

        Returns:
            Dict: roc_auc, pr_auc, best_threshold, best_f1
        """
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

        # ============================================================
        # TODO(human): ROC-AUC, PR-AUC 계산
        # ============================================================
        # ROC-AUC: roc_auc_score(y_true, scores)
        # PR-AUC: precision_recall_curve -> auc
        #
        # Hint: sklearn.metrics 사용

        roc_auc = roc_auc_score(y_true, scores)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall_curve, precision_curve)

        if roc_auc is None:
            raise NotImplementedError(
                "TODO(human): AUCMetrics.compute()를 구현해주세요!"
            )

        # Find best threshold using F1
        best_f1, best_threshold = self._find_best_threshold(y_true, scores)

        return {
            'metric_name': 'AUCMetrics',
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'best_threshold': float(best_threshold),
            'best_f1': float(best_f1)
        }

    def _find_best_threshold(self, y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
        """
        Find the threshold that maximizes F1 score

        Returns:
            Tuple[best_f1, best_threshold]
        """
        from sklearn.metrics import precision_recall_curve, f1_score

        precision, recall, thresholds = precision_recall_curve(y_true, scores)

        # Calculate F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Find best
        best_idx = np.argmax(f1_scores[:-1])  # last value is for threshold=max
        best_f1 = f1_scores[best_idx]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

        return best_f1, best_threshold


class RangeBasedMetrics(BaseMetric):
    """
    Range-based Precision/Recall (for time series)

    Unlike point-wise metrics, this evaluates based on overlapping ranges.
    Better reflects the practical value of anomaly detection.

    Reference:
    - Tatbul et al., "Precision and Recall for Time Series" (NeurIPS 2018)
    """

    def __init__(self, alpha: float = 0.0, cardinality: str = 'reciprocal', bias: str = 'flat'):
        """
        Args:
            alpha: Weight for existence reward (0 = pure overlap)
            cardinality: How to handle multiple predictions per ground truth
                        ('one', 'reciprocal', 'udf_gamma')
            bias: Position bias ('flat', 'front', 'middle', 'back')
        """
        super().__init__()
        self.alpha = alpha
        self.cardinality = cardinality
        self.bias = bias

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute range-based precision and recall

        Args:
            y_true: Ground truth (0: normal, 1: anomaly)
            y_pred: Predictions (0: normal, 1: anomaly)

        Returns:
            Dict: range_precision, range_recall, range_f1
        """
        self._validate_inputs(y_true, y_pred)

        # Find segments
        true_segments = self._find_segments(y_true)
        pred_segments = self._find_segments(y_pred)

        if len(true_segments) == 0:
            # No ground truth anomalies
            return {
                'metric_name': 'RangeBasedMetrics',
                'range_precision': 0.0 if len(pred_segments) > 0 else 1.0,
                'range_recall': 1.0,
                'range_f1': 0.0 if len(pred_segments) > 0 else 1.0
            }

        if len(pred_segments) == 0:
            # No predictions
            return {
                'metric_name': 'RangeBasedMetrics',
                'range_precision': 1.0,
                'range_recall': 0.0,
                'range_f1': 0.0
            }

        # Compute range-based recall
        range_recall = self._compute_range_recall(true_segments, pred_segments, len(y_true))

        # Compute range-based precision (swap roles)
        range_precision = self._compute_range_recall(pred_segments, true_segments, len(y_true))

        # F1
        if range_precision + range_recall > 0:
            range_f1 = 2 * range_precision * range_recall / (range_precision + range_recall)
        else:
            range_f1 = 0.0

        return {
            'metric_name': 'RangeBasedMetrics',
            'range_precision': float(range_precision),
            'range_recall': float(range_recall),
            'range_f1': float(range_f1)
        }

    def _find_segments(self, y: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments of 1s"""
        segments = []
        in_segment = False
        start = 0

        for i, val in enumerate(y):
            if val == 1 and not in_segment:
                start = i
                in_segment = True
            elif val == 0 and in_segment:
                segments.append((start, i))
                in_segment = False

        if in_segment:
            segments.append((start, len(y)))

        return segments

    def _compute_range_recall(
        self,
        true_segments: List[Tuple[int, int]],
        pred_segments: List[Tuple[int, int]],
        total_len: int
    ) -> float:
        """Compute range-based recall"""
        recalls = []

        for t_start, t_end in true_segments:
            t_len = t_end - t_start

            # Find overlapping predictions
            overlap = 0
            for p_start, p_end in pred_segments:
                # Calculate overlap
                o_start = max(t_start, p_start)
                o_end = min(t_end, p_end)
                if o_end > o_start:
                    overlap += (o_end - o_start)

            # Recall for this segment
            segment_recall = overlap / t_len if t_len > 0 else 0
            recalls.append(segment_recall)

        return np.mean(recalls) if recalls else 0.0


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray = None) -> Dict:
    """
    Compute all metrics at once

    Args:
        y_true: Ground truth labels
        y_pred: Binary predictions
        scores: Anomaly scores (optional, for AUC metrics)

    Returns:
        Dict with all metric results
    """
    point_f1 = PointF1()
    pa_f1 = PointAdjustF1()
    range_based = RangeBasedMetrics()

    results = {
        'point_f1': point_f1.compute(y_true, y_pred),
        'pa_f1': pa_f1.compute(y_true, y_pred),
        'range_based': range_based.compute(y_true, y_pred)
    }

    # Add AUC metrics if scores provided
    if scores is not None:
        auc_metrics = AUCMetrics()
        results['auc'] = auc_metrics.compute(y_true, scores)

    return results
