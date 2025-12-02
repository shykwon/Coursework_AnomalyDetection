# -*- coding: utf-8 -*-
"""
TSAD 파이프라인
End-to-End 이상치 탐지 파이프라인
"""

from typing import Dict, Optional, List, Tuple, Any

import numpy as np

from preprocessing.base import BasePreprocessor
from models.base import BaseModel


class TSADPipeline:
    """
    Time Series Anomaly Detection Pipeline

    🎓 [학습 포인트]
    파이프라인 흐름:
    1. Preprocessing: 데이터 스케일링, 스무딩 등
    2. Model: 이상치 점수 계산
    3. Postprocessing: 임계값 적용, 라벨링

    Attributes:
        preprocessor: 전처리기 (BasePreprocessor)
        model: 이상치 탐지 모델 (BaseModel)
        threshold: 이상치 판정 임계값
    """

    def __init__(
        self,
        preprocessor: Optional[BasePreprocessor] = None,
        model: Optional[BaseModel] = None,
        threshold: Optional[float] = None,
    ):
        """
        Args:
            preprocessor: 전처리기 (None이면 스킵)
            model: 이상치 탐지 모델 (필수)
            threshold: 이상치 판정 임계값 (None이면 자동 설정)
        """
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold

        self._is_fitted = False
        self._train_scores: Optional[np.ndarray] = None

    def fit(self, train_data: np.ndarray, **kwargs) -> 'TSADPipeline':
        """
        파이프라인 학습

        Args:
            train_data: 학습 데이터 (n_samples, n_features)

        Returns:
            self
        """
        if self.model is None:
            raise ValueError("모델이 설정되지 않았습니다.")

        # ============================================================
        # TODO(human): 파이프라인 fit 로직 구현
        # ============================================================
        # 1. preprocessor가 있으면: fit_transform 적용
        # 2. model: fit 수행
        # 3. threshold가 None이면: 학습 데이터의 이상치 점수 기반으로 자동 설정
        #
        # Hint:
        # - self.preprocessor.fit_transform(train_data)
        # - self.model.fit(processed_data)
        # - 자동 threshold = mean + 3 * std (3-sigma rule)

        processed_data = train_data

        # Step 1: 전처리
        if self.preprocessor is not None:
            processed_data = self.preprocessor.fit_transform(train_data)

            if processed_data is None:
                raise NotImplementedError(
                    "TODO(human): preprocessor.fit_transform()을 호출해주세요!"
                )

        # Step 2: 모델 학습
        self.model.fit(processed_data, **kwargs)

        # Step 3: 자동 threshold 설정
        if self.threshold is None:
            self._train_scores = self.model.get_anomaly_score(processed_data)
            # 유효한 점수만 사용 (앞부분 패딩 제외)
            valid_scores = self._train_scores[self._train_scores > 0]

            if len(valid_scores) > 0:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                self.threshold = mean_score + 3 * std_score  # 3-sigma rule
                print(f"자동 threshold 설정: {self.threshold:.6f}")

        self._is_fitted = True
        return self

    def predict(self, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상치 탐지 수행

        Args:
            test_data: 테스트 데이터 (n_samples, n_features)

        Returns:
            Tuple[이상치 점수, 이상치 라벨]
            - scores: (n_samples,) 이상치 점수
            - labels: (n_samples,) 0=정상, 1=이상치
        """
        if not self._is_fitted:
            raise RuntimeError("파이프라인이 fit되지 않았습니다.")

        # ============================================================
        # TODO(human): 파이프라인 predict 로직 구현
        # ============================================================
        # 1. preprocessor가 있으면: transform 적용 (fit 아님!)
        # 2. model: get_anomaly_score로 점수 계산
        # 3. threshold 기준으로 라벨 생성

        processed_data = test_data

        # Step 1: 전처리
        if self.preprocessor is not None:
            processed_data = self.preprocessor.transform(test_data)


            if processed_data is None:
                raise NotImplementedError(
                    "TODO(human): preprocessor.transform()을 호출해주세요!"
                )

        # Step 2: 이상치 점수 계산
        scores = self.model.get_anomaly_score(processed_data)

        # Step 3: 라벨 생성
        labels = (scores > self.threshold).astype(np.int32)

        if labels is None:
            raise NotImplementedError(
                "TODO(human): threshold 기준 라벨 생성을 구현해주세요!"
            )

        return scores, labels

    def fit_predict(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 후 예측 수행

        Args:
            train_data: 학습 데이터
            test_data: 테스트 데이터

        Returns:
            Tuple[이상치 점수, 이상치 라벨]
        """
        self.fit(train_data, **kwargs)
        return self.predict(test_data)

    def get_threshold(self) -> float:
        """현재 threshold 반환"""
        return self.threshold

    def set_threshold(self, threshold: float):
        """threshold 수동 설정"""
        self.threshold = threshold
