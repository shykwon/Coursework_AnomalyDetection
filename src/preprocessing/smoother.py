# -*- coding: utf-8 -*-
"""
Smoothing 전처리 모듈
노이즈 제거를 위한 스무딩 기법 구현
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import BasePreprocessor


class EWMASmoother(BasePreprocessor):
    """
    Exponentially Weighted Moving Average (EWMA) Smoother

    EWMA는 최근 데이터에 더 높은 가중치를 부여하는 이동 평균 기법입니다.
    노이즈가 많은 시계열 데이터를 부드럽게 만들어 이상 탐지 성능을 향상시킵니다.

    🎓 [학습 포인트]
    - EWMA 공식: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
    - alpha = 2 / (span + 1), span이 클수록 더 smooth
    - 장점: 최근 데이터에 더 민감, 계산 효율적
    - 단점: 급격한 변화(이상치)를 놓칠 수 있음

    Attributes:
        span: EWMA window size (클수록 더 smooth)
        alpha: 감쇠 계수 (2 / (span + 1))
    """

    def __init__(self, span: int = 10):
        """
        Args:
            span: EWMA window size (default: 10)
                  - 작은 값: 원본에 가까움, 노이즈에 민감
                  - 큰 값: 더 smooth, 급격한 변화 놓칠 수 있음
        """
        super().__init__()
        self.span = span
        self.alpha = 2.0 / (span + 1)

        # 역변환을 위한 원본 저장 (옵션)
        self._original_data: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'EWMASmoother':
        """
        Smoother는 학습할 파라미터가 없으므로 단순히 상태만 설정

        Args:
            data: 학습 데이터 (n_samples, n_features)

        Returns:
            self
        """
        # EWMA는 학습 파라미터가 없지만, fit 호출 확인을 위해 플래그 설정
        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        EWMA 스무딩 적용

        Args:
            data: 변환할 데이터 (n_samples, n_features)

        Returns:
            스무딩된 데이터
        """
        self._check_is_fitted()

        # 원본 저장 (역변환용)
        self._original_data = data.copy()

        # ============================================================
        # TODO(human): EWMA 스무딩 구현
        # ============================================================
        # EWMA (Exponentially Weighted Moving Average) 공식:
        #   y_t = alpha * x_t + (1 - alpha) * y_{t-1}
        #
        # pandas의 ewm().mean() 사용 가능:7
        #   pd.DataFrame(data).ewm(span=self.span).mean().values
        #
        # 또는 직접 구현:
        #   for t in range(1, len(data)):
        #       smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t-1]

        smoothed = pd.DataFrame(data).ewm(span=self.span, adjust=False).mean().values
        
        if smoothed is None:
            raise NotImplementedError(
                "TODO(human): EWMASmoother.transform()의 EWMA 스무딩을 구현해주세요!"
            )

        return smoothed

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        EWMA 역변환

        Note: EWMA는 정보 손실이 있어 완벽한 역변환이 불가능합니다.
        여기서는 원본 데이터를 저장해두고 반환하거나,
        근사적인 역변환을 수행합니다.

        Args:
            data: 역변환할 데이터

        Returns:
            근사 역변환된 데이터
        """
        self._check_is_fitted()

        # 원본 데이터가 있으면 반환 (같은 데이터에 대한 역변환인 경우)
        if self._original_data is not None and len(data) == len(self._original_data):
            return self._original_data

        # 근사 역변환: smoothed 데이터에서 원본 추정
        # y_t = alpha * x_t + (1 - alpha) * y_{t-1}
        # x_t = (y_t - (1 - alpha) * y_{t-1}) / alpha
        original_approx = np.zeros_like(data)
        original_approx[0] = data[0]

        for t in range(1, len(data)):
            original_approx[t] = (data[t] - (1 - self.alpha) * data[t-1]) / self.alpha

        return original_approx


class MovingAverageSmoother(BasePreprocessor):
    """
    Simple Moving Average (SMA) Smoother

    단순 이동 평균으로 데이터를 스무딩합니다.

    Attributes:
        window: 이동 평균 윈도우 크기
    """

    def __init__(self, window: int = 5):
        """
        Args:
            window: 이동 평균 윈도우 크기
        """
        super().__init__()
        self.window = window
        self._original_data: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'MovingAverageSmoother':
        """SMA는 학습할 파라미터가 없음"""
        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Simple Moving Average 적용

        Args:
            data: 변환할 데이터 (n_samples, n_features)

        Returns:
            스무딩된 데이터
        """
        self._check_is_fitted()
        self._original_data = data.copy()

        df = pd.DataFrame(data)
        # min_periods=1로 초기 NaN 방지
        smoothed = df.rolling(window=self.window, min_periods=1, center=False).mean()

        return smoothed.values

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """SMA 역변환 (원본 반환)"""
        self._check_is_fitted()

        if self._original_data is not None and len(data) == len(self._original_data):
            return self._original_data

        # 완벽한 역변환 불가능
        return data
