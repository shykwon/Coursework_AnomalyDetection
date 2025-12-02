# -*- coding: utf-8 -*-
"""
Threshold 후처리 모듈
이상 점수를 이진 레이블로 변환하는 임계값 기법
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseThreshold(ABC):
    """
    임계값 추상 베이스 클래스

    이상 점수(anomaly score)를 이진 레이블(0/1)로 변환합니다.

    🎓 [학습 포인트]
    - 임계값 선정은 이상 탐지의 핵심 도전 과제
    - 너무 낮으면 FP 증가, 너무 높으면 FN 증가
    - 다양한 전략: 고정(sigma), 백분위수, 동적(EWMA), 극단값 이론(POT)
    """

    def __init__(self):
        self._is_fitted = False
        self.threshold_: Optional[float] = None

    @abstractmethod
    def fit(self, scores: np.ndarray) -> 'BaseThreshold':
        """
        학습 데이터의 이상 점수에서 임계값 학습

        Args:
            scores: 이상 점수 배열 (n_samples,) 또는 (n_samples, 1)

        Returns:
            self
        """
        pass

    @abstractmethod
    def apply(self, scores: np.ndarray) -> np.ndarray:
        """
        임계값 적용하여 이진 레이블 반환

        Args:
            scores: 이상 점수 배열

        Returns:
            이진 레이블 (0: 정상, 1: 이상)
        """
        pass

    def fit_apply(self, scores: np.ndarray) -> np.ndarray:
        """fit과 apply를 한 번에 수행"""
        return self.fit(scores).apply(scores)

    def _check_is_fitted(self):
        """fit 여부 확인"""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}이(가) fit되지 않았습니다. "
                "먼저 fit()을 호출하세요."
            )


class FixedThreshold(BaseThreshold):
    """
    고정 임계값 (Fixed Threshold)

    두 가지 방법을 지원:
    1. Sigma 기반: threshold = mean + n_sigma * std
    2. Percentile 기반: threshold = percentile(scores, percentile)

    🎓 [학습 포인트]
    - Sigma 방법: 정규분포 가정, 3-sigma는 99.7% 신뢰구간
    - Percentile 방법: 분포 무관, 상위 k%를 이상으로 판정
    - 실제로는 이상 점수가 정규분포를 따르지 않는 경우가 많음

    Attributes:
        method: 'sigma' 또는 'percentile'
        n_sigma: sigma 방법 시 표준편차 배수 (기본: 3.0)
        percentile: percentile 방법 시 백분위수 (기본: 95.0)
    """

    def __init__(
        self,
        method: str = 'sigma',
        n_sigma: float = 3.0,
        percentile: float = 95.0
    ):
        """
        Args:
            method: 'sigma' 또는 'percentile'
            n_sigma: sigma 방법 시 표준편차 배수
            percentile: percentile 방법 시 백분위수 (0-100)
        """
        super().__init__()
        self.method = method
        self.n_sigma = n_sigma
        self.percentile = percentile

        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, scores: np.ndarray) -> 'FixedThreshold':
        """
        이상 점수에서 임계값 학습

        Args:
            scores: 이상 점수 배열

        Returns:
            self
        """
        scores = np.asarray(scores).flatten()

        if self.method == 'sigma':
   

            self.mean_ = np.mean(scores)
            self.std_ = np.std(scores)
            self.threshold_ = self.mean_ + self.n_sigma * self.std_


            if self.threshold_ is None:
                raise NotImplementedError(
                    "TODO(human): FixedThreshold.fit()의 sigma 기반 임계값을 구현해주세요!"
                )

        elif self.method == 'percentile':
            # Percentile 기반 임계값
            self.threshold_ = np.percentile(scores, self.percentile)

        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'sigma' or 'percentile'.")

        self._is_fitted = True
        return self

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """
        임계값 적용하여 이진 레이블 반환

        Args:
            scores: 이상 점수 배열

        Returns:
            이진 레이블 (0: 정상, 1: 이상)
        """
        self._check_is_fitted()
        scores = np.asarray(scores).flatten()

        # 임계값 초과 시 이상(1), 이하 시 정상(0)
        predictions = (scores > self.threshold_).astype(int)

        return predictions


class EWMAThreshold(BaseThreshold):
    """
    EWMA 기반 동적 임계값

    시계열의 로컬 통계량을 사용하여 동적으로 임계값을 조정합니다.
    threshold_t = ewma_mean_t + n_sigma * ewma_std_t

    🎓 [학습 포인트]
    - 시계열 데이터는 시간에 따라 분포가 변할 수 있음 (non-stationary)
    - 동적 임계값은 이러한 변화에 적응
    - 장점: 분포 변화에 강건, 단점: 계산 복잡도 증가

    Attributes:
        span: EWMA window size
        n_sigma: 표준편차 배수
    """

    def __init__(self, span: int = 20, n_sigma: float = 3.0):
        """
        Args:
            span: EWMA window size
            n_sigma: 표준편차 배수
        """
        super().__init__()
        self.span = span
        self.n_sigma = n_sigma

        self._ewma_mean: Optional[np.ndarray] = None
        self._ewma_std: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray) -> 'EWMAThreshold':
        """
        EWMA 기반 동적 통계량 계산

        Args:
            scores: 이상 점수 배열

        Returns:
            self
        """
        import pandas as pd

        scores = np.asarray(scores).flatten()
        series = pd.Series(scores)

        # EWMA mean
        self._ewma_mean = series.ewm(span=self.span, adjust=False).mean().values

        # EWMA std (rolling std with ewm weighting approximation)
        # 정확한 EWMA std는 복잡하므로 rolling std 사용
        self._ewma_std = series.rolling(window=self.span, min_periods=1).std().fillna(series.std()).values

        # 동적 임계값 계산
        self.threshold_ = self._ewma_mean + self.n_sigma * self._ewma_std

        self._is_fitted = True
        return self

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """
        동적 임계값 적용

        Args:
            scores: 이상 점수 배열

        Returns:
            이진 레이블
        """
        self._check_is_fitted()
        scores = np.asarray(scores).flatten()

        # 학습 시 계산한 동적 임계값과 비교
        if len(scores) == len(self.threshold_):
            predictions = (scores > self.threshold_).astype(int)
        else:
            # 새로운 데이터에 대해서는 마지막 임계값 사용
            last_threshold = self.threshold_[-1] if len(self.threshold_) > 0 else 0
            predictions = (scores > last_threshold).astype(int)

        return predictions


class AdaptiveThreshold(BaseThreshold):
    """
    적응형 임계값 (Best F1 기반)

    Ground truth가 있을 때, 최적의 F1-score를 주는 임계값을 찾습니다.

    🎓 [학습 포인트]
    - 실제 환경에서는 레이블이 없어서 사용 불가
    - 모델 평가 시 upper bound로 활용
    - 다양한 임계값을 시도하여 최적값 탐색

    Attributes:
        n_thresholds: 탐색할 임계값 개수
    """

    def __init__(self, n_thresholds: int = 100):
        """
        Args:
            n_thresholds: 탐색할 임계값 개수
        """
        super().__init__()
        self.n_thresholds = n_thresholds
        self.best_f1_: Optional[float] = None

    def fit(self, scores: np.ndarray, y_true: Optional[np.ndarray] = None) -> 'AdaptiveThreshold':
        """
        최적 임계값 탐색

        Args:
            scores: 이상 점수 배열
            y_true: Ground truth 레이블 (필수)

        Returns:
            self
        """
        if y_true is None:
            raise ValueError("AdaptiveThreshold requires y_true for fitting")

        scores = np.asarray(scores).flatten()
        y_true = np.asarray(y_true).flatten()

        # 임계값 후보 생성
        min_score, max_score = scores.min(), scores.max()
        thresholds = np.linspace(min_score, max_score, self.n_thresholds)

        best_f1 = 0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)

            # F1 계산
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold_ = best_threshold
        self.best_f1_ = best_f1
        self._is_fitted = True

        return self

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """최적 임계값 적용"""
        self._check_is_fitted()
        scores = np.asarray(scores).flatten()
        return (scores > self.threshold_).astype(int)
