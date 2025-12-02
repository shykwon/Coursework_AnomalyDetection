# -*- coding: utf-8 -*-
"""
스케일러 모듈
MinMax, Standard 스케일링 구현
"""

from typing import Literal, Optional

import numpy as np

from .base import BasePreprocessor


class Scaler(BasePreprocessor):
    """
    데이터 스케일링 클래스

    🎓 [학습 포인트]
    - MinMax: (x - min) / (max - min) → [0, 1] 범위로 변환
    - Standard: (x - mean) / std → 평균 0, 분산 1로 변환

    Attributes:
        method: 스케일링 방법 ('minmax' 또는 'standard')
    """

    def __init__(self, method: Literal['minmax', 'standard'] = 'standard'):
        """
        Args:
            method: 스케일링 방법
                - 'minmax': Min-Max 정규화 [0, 1]
                - 'standard': Z-score 표준화 (평균 0, 표준편차 1)
        """
        super().__init__()
        if method not in ['minmax', 'standard']:
            raise ValueError(f"method는 'minmax' 또는 'standard'여야 합니다. 받은 값: {method}")

        self.method = method

        # 학습된 파라미터 (fit에서 설정)
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'Scaler':
        """
        스케일링 파라미터 학습

        Args:
            data: 학습 데이터 (n_samples, n_features)

        Returns:
            self
        """
        # ============================================================
        # TODO(human): 스케일링 파라미터 계산
        # ============================================================
        # MinMax: 각 feature의 min, max 계산
        # Standard: 각 feature의 mean, std 계산
        #
        # Hint: np.min(data, axis=0), np.max(data, axis=0)
        #       np.mean(data, axis=0), np.std(data, axis=0)
        # axis=0은 각 컬럼(feature)별로 계산한다는 의미

        if self.method == 'minmax':
            self._min = np.min(data, axis=0)
            self._max = np.max(data, axis=0)

        else:  # standard
            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0)

        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 스케일링 적용

        Args:
            data: 변환할 데이터 (n_samples, n_features)

        Returns:
            스케일링된 데이터
        """
        self._check_is_fitted()

        # ============================================================
        # TODO(human): 스케일링 변환 구현
        # ============================================================
        # MinMax: (x - min) / (max - min)
        # Standard: (x - mean) / std
        #
        # 주의: 0으로 나누기 방지 필요 (eps=1e-8 사용)

        eps = 1e-8  # 0으로 나누기 방지

        if self.method == 'minmax':
            scaled = (data - self._min) / (self._max - self._min + eps)

        else:  # standard
            scaled = (data - self._mean) / (self._std + eps)

        if scaled is None:
            raise NotImplementedError(
                "TODO(human): Scaler.transform()의 스케일링 변환을 구현해주세요!"
            )

        return scaled

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        역변환 (원본 스케일 복원)

        Args:
            data: 역변환할 데이터

        Returns:
            원본 스케일의 데이터
        """
        self._check_is_fitted()

        # ============================================================
        # TODO(human): 역변환 구현
        # ============================================================
        # MinMax: x * (max - min) + min
        # Standard: x * std + mean

        eps = 1e-8

        if self.method == 'minmax':
            original = data * (self._max - self._min + eps) + self._min

        else:  # standard
            original = data * (self._std + eps) + self._mean

        if original is None:
            raise NotImplementedError(
                "TODO(human): Scaler.inverse_transform()의 역변환을 구현해주세요!"
            )

        return original

    def get_params(self) -> dict:
        """학습된 파라미터 반환"""
        self._check_is_fitted()

        if self.method == 'minmax':
            return {'min': self._min, 'max': self._max}
        else:
            return {'mean': self._mean, 'std': self._std}
