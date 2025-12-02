# -*- coding: utf-8 -*-
"""
모델 기본 클래스
모든 이상치 탐지 모델의 추상 베이스 클래스 정의
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseModel(ABC):
    """
    이상치 탐지 모델 추상 베이스 클래스

    모든 모델은 이 클래스를 상속받아 구현합니다.

    🎓 [학습 포인트]
    - fit(): 학습 데이터로 모델 훈련
    - predict(): 테스트 데이터에서 예측값 반환
    - get_anomaly_score(): 이상치 점수 계산 (핵심!)
        - Prediction-based: |실제값 - 예측값| (reconstruction error)
        - Reconstruction-based: -log_prob 또는 reconstruction error
    """

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, train_data: np.ndarray, **kwargs) -> 'BaseModel':
        """
        모델 학습

        Args:
            train_data: 학습 데이터 (n_samples, n_features)
            **kwargs: 추가 학습 파라미터

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        예측 수행

        Args:
            data: 입력 데이터 (n_samples, n_features)

        Returns:
            예측값 (n_samples, n_features)
        """
        pass

    @abstractmethod
    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """
        이상치 점수 계산

        Args:
            data: 입력 데이터 (n_samples, n_features)

        Returns:
            이상치 점수 (n_samples,) - 값이 클수록 이상치일 가능성 높음
        """
        pass

    def fit_predict(self, train_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 후 예측 수행

        Args:
            train_data: 학습 데이터
            test_data: 테스트 데이터

        Returns:
            Tuple[예측값, 이상치 점수]
        """
        self.fit(train_data)
        predictions = self.predict(test_data)
        scores = self.get_anomaly_score(test_data)
        return predictions, scores

    def _check_is_fitted(self):
        """fit 여부 확인"""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__}이(가) fit되지 않았습니다. "
                "먼저 fit()을 호출하세요."
            )
