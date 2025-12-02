# -*- coding: utf-8 -*-
"""
전처리 기본 클래스
모든 전처리기의 추상 베이스 클래스 정의
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BasePreprocessor(ABC):
    """
    전처리기 추상 베이스 클래스

    모든 전처리기는 이 클래스를 상속받아 구현합니다.
    sklearn의 fit/transform 패턴을 따릅니다.

    🎓 [학습 포인트]
    - fit(): 학습 데이터에서 파라미터 학습 (예: mean, std)
    - transform(): 학습된 파라미터로 데이터 변환
    - inverse_transform(): 원본 스케일로 복원
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BasePreprocessor':
        """
        학습 데이터에서 전처리 파라미터 학습

        Args:
            data: 학습 데이터 (n_samples, n_features)

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 변환

        Args:
            data: 변환할 데이터 (n_samples, n_features)

        Returns:
            변환된 데이터
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        역변환 (원본 스케일 복원)

        Args:
            data: 역변환할 데이터 (n_samples, n_features)

        Returns:
            원본 스케일의 데이터
        """
        pass

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        fit과 transform을 한 번에 수행

        Args:
            data: 학습 및 변환할 데이터

        Returns:
            변환된 데이터
        """
        return self.fit(data).transform(data)

    def _check_is_fitted(self):
        """fit 여부 확인"""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}이(가) fit되지 않았습니다. "
                "먼저 fit() 또는 fit_transform()을 호출하세요."
            )
