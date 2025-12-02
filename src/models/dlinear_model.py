# -*- coding: utf-8 -*-
"""
DLinear 모델 Wrapper
Prediction-based 이상치 탐지 모델

🎓 [학습 포인트]
Prediction-based 접근법:
1. 정상 데이터로 "다음 값 예측" 모델 학습
2. 테스트 시 예측값과 실제값의 차이(오차)를 이상치 점수로 사용
3. 이상치 = 예측하기 어려운 패턴 = 큰 예측 오차
"""

from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel
from .cores.dlinear import DLinear, DLinearConfig


class DLinearModel(BaseModel):
    """
    DLinear 기반 이상치 탐지 모델

    Attributes:
        seq_len: 입력 시퀀스 길이 (lookback window)
        pred_len: 예측 길이 (기본 1)
        lr: 학습률
        epochs: 학습 에폭 수
        batch_size: 배치 크기
        device: 'cuda' 또는 'cpu'
    """

    def __init__(
        self,
        seq_len: int = 100,
        pred_len: int = 1,
        lr: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        individual: bool = False,
        device: Optional[str] = None,
        early_stopping: bool = True,
        patience: int = 3,
        val_ratio: float = 0.1,
    ):
        """
        Args:
            seq_len: 입력 시퀀스 길이 (lookback window)
            pred_len: 예측 길이 (기본 1)
            lr: 학습률
            epochs: 최대 학습 에폭 수
            batch_size: 배치 크기
            individual: 변수별 독립 학습 여부
            device: 'cuda' 또는 'cpu'
            early_stopping: Early Stopping 활성화 여부
            patience: Early Stopping patience (연속 n번 개선 없으면 종료)
            val_ratio: Validation 데이터 비율
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.individual = individual

        # Early Stopping 설정
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_ratio = val_ratio

        # Device 설정
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model: Optional[DLinear] = None
        self.n_features: Optional[int] = None
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[dict] = None

    def _create_sequences(self, data: np.ndarray) -> tuple:
        """
        슬라이딩 윈도우로 시퀀스 생성

        Args:
            data: (n_samples, n_features)

        Returns:
            X: (n_sequences, seq_len, n_features)
            y: (n_sequences, pred_len, n_features)
        """
        # ============================================================
        # TODO(human): 슬라이딩 윈도우 시퀀스 생성
        # ============================================================
        # 시계열을 (입력, 타겟) 쌍으로 변환
        # 예: seq_len=3, pred_len=1일 때
        #     data = [0,1,2,3,4,5]
        #     X[0] = [0,1,2], y[0] = [3]
        #     X[1] = [1,2,3], y[1] = [4]
        #     ...
        #
        # Hint: 반복문으로 윈도우를 이동하며 슬라이싱

        X, y = [], []

        for i in range(len(data) - self.seq_len - self.pred_len + 1):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len:i + self.seq_len + self.pred_len])

        if len(X) == 0:
            raise NotImplementedError(
                "TODO(human): _create_sequences()의 슬라이딩 윈도우를 구현해주세요!"
            )

        return np.array(X), np.array(y)

    def fit(self, train_data: np.ndarray, verbose: bool = True, **kwargs) -> 'DLinearModel':
        """
        DLinear 모델 학습 (Early Stopping 지원)

        Args:
            train_data: 학습 데이터 (n_samples, n_features)
            verbose: 학습 로그 출력 여부
        """
        self.n_features = train_data.shape[1]

        # 모델 생성
        config = DLinearConfig(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=self.n_features,
            individual=self.individual,
        )
        self.model = DLinear(config).to(self.device)

        # 시퀀스 생성
        X, y = self._create_sequences(train_data)

        # Train/Validation 분할 (Early Stopping용)
        if self.early_stopping and self.val_ratio > 0:
            n_val = int(len(X) * self.val_ratio)
            X_train, X_val = X[:-n_val], X[-n_val:]
            y_train, y_val = y[:-n_val], y[-n_val:]

            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        else:
            X_train_tensor = torch.FloatTensor(X).to(self.device)
            y_train_tensor = torch.FloatTensor(y).to(self.device)
            X_val_tensor, y_val_tensor = None, None

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 학습
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Early Stopping 변수
        self.best_val_loss = float('inf')
        self.best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation (Early Stopping)
            if self.early_stopping and X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_output = self.model(X_val_tensor)
                    val_loss = criterion(val_output, y_val_tensor).item()

                # Best model 저장
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] Train: {avg_train_loss:.6f}, Val: {val_loss:.6f}")

                # Early Stopping 체크
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  ⚡ Early stopping at epoch {epoch+1} (patience={self.patience})")
                    break
            else:
                if verbose and (epoch + 1) % max(1, self.epochs // 5) == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {avg_train_loss:.6f}")

        # Best model 복원
        if self.early_stopping and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"  ✓ Best model restored (val_loss: {self.best_val_loss:.6f})")

        self._is_fitted = True
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        예측 수행

        Args:
            data: 입력 데이터 (n_samples, n_features)

        Returns:
            예측값 (n_valid_samples, pred_len, n_features)
        """
        self._check_is_fitted()

        X, _ = self._create_sequences(data)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)

        return predictions.cpu().numpy()

    def get_anomaly_score(self, data: np.ndarray) -> np.ndarray:
        """
        이상치 점수 계산

        🎓 [학습 포인트]
        Prediction-based 이상치 점수:
        score = |실제값 - 예측값|의 평균 (각 시점별)

        Args:
            data: 입력 데이터 (n_samples, n_features)

        Returns:
            이상치 점수 (n_samples,)
        """
        self._check_is_fitted()

        X, y = self._create_sequences(data)

        # 배치 단위로 추론 (메모리 효율)
        self.model.eval()
        batch_size = 1024
        predictions_list = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                batch_pred = self.model(batch_X).cpu().numpy()
                predictions_list.append(batch_pred)
                del batch_X  # GPU 메모리 해제
                torch.cuda.empty_cache()

        predictions = np.concatenate(predictions_list, axis=0)

        # ============================================================
        # TODO(human): 이상치 점수 계산
        # ============================================================
        # predictions: (n_sequences, pred_len, n_features)
        # y: (n_sequences, pred_len, n_features)
        #
        # 이상치 점수 = |y - predictions|의 feature 평균
        # 결과 shape: (n_sequences,) 또는 (n_sequences, pred_len)
        #
        # Hint: np.abs(), np.mean(axis=...)

        # 예측 오차 계산 (MAE)
        errors = np.abs(y - predictions)  # (n_seq, pred_len, n_features)

        scores = scores = np.mean(errors, axis=(1, 2))  # (n_sequences,)

        if scores is None:
            raise NotImplementedError(
                "TODO(human): get_anomaly_score()의 점수 계산을 구현해주세요!"
            )

        # 전체 데이터 길이에 맞게 패딩 (앞부분은 점수 계산 불가)
        full_scores = np.zeros(len(data))
        full_scores[self.seq_len:self.seq_len + len(scores)] = scores

        return full_scores
