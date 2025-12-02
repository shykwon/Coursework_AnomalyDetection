# -*- coding: utf-8 -*-
"""
Detrending 전처리 모듈
시계열 데이터에서 추세(Trend) 성분 제거
"""

from typing import Optional

import numpy as np
import pandas as pd

from .base import BasePreprocessor


class MovingAverageDetrender(BasePreprocessor):
    """
    이동 평균 기반 Detrending

    원본 시계열에서 이동 평균(Trend)을 빼서 추세를 제거합니다.
    detrended = original - moving_average(original)

    🎓 [학습 포인트]
    - Trend 제거 이유: 추세가 있으면 정상성(Stationarity) 위반
    - 이동 평균: 장기적 추세를 포착
    - 잔차(Residual): 단기 변동 + 노이즈 + 이상치
    - 이상 탐지에서는 잔차에서 이상치를 찾는 것이 더 효과적

    Attributes:
        window: 이동 평균 윈도우 크기 (주기에 맞춤, 예: 24시간)
        trend_: 학습된 추세 성분
    """

    def __init__(self, window: int = 24):
        """
        Args:
            window: 이동 평균 윈도우 크기
                   - 일반적으로 데이터의 주기(period)에 맞춤
                   - 예: 시간별 데이터에서 일별 추세 제거 → window=24
        """
        super().__init__()
        self.window = window
        self.trend_: Optional[np.ndarray] = None
        self._last_trend_values: Optional[np.ndarray] = None  # 마지막 window만큼의 값

    def fit(self, data: np.ndarray) -> 'MovingAverageDetrender':
        """
        학습 데이터에서 추세 패턴 학습

        Args:
            data: 학습 데이터 (n_samples, n_features)

        Returns:
            self
        """
        df = pd.DataFrame(data)

        # 이동 평균으로 추세 계산
        self.trend_ = df.rolling(window=self.window, min_periods=1, center=False).mean().values

        # 마지막 window 크기만큼의 평균값 저장 (테스트 데이터 적용 시 사용)
        self._last_trend_values = data[-self.window:].mean(axis=0) if len(data) >= self.window else data.mean(axis=0)

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        추세 제거 (Detrending)

        Args:
            data: 변환할 데이터 (n_samples, n_features)

        Returns:
            추세가 제거된 데이터
        """
        self._check_is_fitted()

        df = pd.DataFrame(data)

        # 이동 평균으로 추세 계산
        trend = df.rolling(window=self.window, min_periods=1, center=False).mean().values

        # 추세 저장 (역변환용)
        self.trend_ = trend

        # Detrending: 원본 - 추세
        detrended = data - trend

        return detrended

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        추세 복원

        Args:
            data: 역변환할 데이터 (detrended)

        Returns:
            추세가 복원된 데이터
        """
        self._check_is_fitted()

        if self.trend_ is not None and len(data) == len(self.trend_):
            # 같은 길이면 저장된 추세 사용
            return data + self.trend_
        else:
            # 다른 길이면 마지막 추세값으로 근사
            return data + self._last_trend_values


class DifferencingDetrender(BasePreprocessor):
    """
    차분(Differencing) 기반 Detrending

    1차 차분: y_t = x_t - x_{t-1}
    추세와 계절성을 제거하는 고전적 방법

    🎓 [학습 포인트]
    - 1차 차분: 선형 추세 제거
    - 계절 차분: x_t - x_{t-period}로 계절성 제거
    - ARIMA 모델의 I(Integrated) 부분이 차분
    """

    def __init__(self, order: int = 1, period: Optional[int] = None):
        """
        Args:
            order: 차분 차수 (1이면 1차 차분)
            period: 계절 차분 주기 (None이면 일반 차분)
        """
        super().__init__()
        self.order = order
        self.period = period
        self._first_values: Optional[np.ndarray] = None  # 역변환용

    def fit(self, data: np.ndarray) -> 'DifferencingDetrender':
        """차분은 학습 파라미터 없음, 첫 값만 저장"""
        if self.period:
            self._first_values = data[:self.period].copy()
        else:
            self._first_values = data[:self.order].copy()

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        차분 적용

        Args:
            data: 변환할 데이터 (n_samples, n_features)

        Returns:
            차분된 데이터
        """
        self._check_is_fitted()

        # 첫 값 저장 (역변환용)
        if self.period:
            self._first_values = data[:self.period].copy()
        else:
            self._first_values = data[:self.order].copy()

        result = data.copy()

        for _ in range(self.order):
            if self.period:
                # 계절 차분
                diff = np.zeros_like(result)
                diff[self.period:] = result[self.period:] - result[:-self.period]
                diff[:self.period] = 0  # 첫 period는 0
                result = diff
            else:
                # 일반 차분
                diff = np.zeros_like(result)
                diff[1:] = result[1:] - result[:-1]
                diff[0] = 0
                result = diff

        return result

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        차분 역변환 (누적합)

        Args:
            data: 차분된 데이터

        Returns:
            원본 복원 (근사)
        """
        self._check_is_fitted()

        result = data.copy()

        for _ in range(self.order):
            if self.period:
                # 계절 역차분
                cumsum = np.zeros_like(result)
                cumsum[:self.period] = self._first_values[:self.period] if self._first_values is not None else 0
                for t in range(self.period, len(result)):
                    cumsum[t] = result[t] + cumsum[t - self.period]
                result = cumsum
            else:
                # 일반 역차분 (누적합)
                cumsum = np.cumsum(result, axis=0)
                if self._first_values is not None:
                    cumsum = cumsum + self._first_values[0]
                result = cumsum

        return result


class STLDetrender(BasePreprocessor):
    """
    STL Decomposition 기반 Detrending

    STL(Seasonal and Trend decomposition using Loess)로 분해 후
    Trend + Seasonal을 제거하고 Residual만 반환

    🎓 [학습 포인트]
    - STL 분해: Original = Trend + Seasonal + Residual
    - Trend: 장기적 추세
    - Seasonal: 주기적 패턴
    - Residual: 나머지 (노이즈 + 이상치)
    - 이상 탐지에서는 Residual에서 이상치를 찾음
    """

    def __init__(self, period: int = 24, robust: bool = True):
        """
        Args:
            period: 계절성 주기
            robust: 이상치에 강건한 STL 사용 여부
        """
        super().__init__()
        self.period = period
        self.robust = robust
        self._trend: Optional[np.ndarray] = None
        self._seasonal: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'STLDetrender':
        """STL은 transform 시점에 분해하므로 fit에서는 플래그만 설정"""
        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        STL 분해 후 Residual 반환

        Args:
            data: 변환할 데이터 (n_samples, n_features)

        Returns:
            Residual (Trend, Seasonal 제거됨)
        """
        from statsmodels.tsa.seasonal import STL

        self._check_is_fitted()

        n_samples, n_features = data.shape
        residuals = np.zeros_like(data)
        self._trend = np.zeros_like(data)
        self._seasonal = np.zeros_like(data)

        for i in range(n_features):
            series = pd.Series(data[:, i])

            # 충분한 데이터가 있을 때만 STL 적용
            if len(series) >= 2 * self.period:
                stl = STL(series, period=self.period, robust=self.robust)
                result = stl.fit()

                self._trend[:, i] = result.trend
                self._seasonal[:, i] = result.seasonal
                residuals[:, i] = result.resid
            else:
                # 데이터가 부족하면 단순 이동평균 사용
                trend = series.rolling(window=self.period, min_periods=1).mean()
                self._trend[:, i] = trend.values
                self._seasonal[:, i] = 0
                residuals[:, i] = data[:, i] - trend.values

        return residuals

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Trend + Seasonal 복원

        Args:
            data: Residual 데이터

        Returns:
            원본 복원 (Trend + Seasonal + Residual)
        """
        self._check_is_fitted()

        if self._trend is not None and self._seasonal is not None:
            return data + self._trend + self._seasonal

        return data
