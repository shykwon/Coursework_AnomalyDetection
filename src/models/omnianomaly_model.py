# -*- coding: utf-8 -*-
"""
OmniAnomaly Model Wrapper

BaseModel 인터페이스를 구현하여 파이프라인에서 사용 가능하게 합니다.

🎓 [학습 포인트]
- Reconstruction-based 이상 탐지: 정상 패턴 복원 학습 → 이상 데이터는 복원 오류 증가
- VAE의 ELBO 손실: Reconstruction + KL Divergence
- 이상 점수: -log p(x|z), 복원 확률이 낮을수록 이상
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseModel
from .cores.omnianomaly_core import OmniAnomalyCore


class OmniAnomaly(BaseModel):
    """
    OmniAnomaly Wrapper (Reconstruction-based)

    원본 논문: "Robust Anomaly Detection for Multivariate Time Series" (KDD 2019)

    핵심 구조:
    - GRU 기반 VAE
    - Planar Normalizing Flow로 사후 분포 표현력 향상
    - 복원 확률 기반 이상 점수

    🎓 [학습 포인트]
    - DLinear(Prediction)와 달리 '복원'으로 이상 탐지
    - 정상 데이터로만 학습 → 이상 데이터는 잘 복원되지 않음
    - Multi-modal 분포도 Normalizing Flow로 표현 가능
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: 모델 설정
                - x_dim: 입력 차원 (feature 수)
                - z_dim: 잠재 공간 차원 (default: 8)
                - hidden_dim: GRU hidden 차원 (default: 100)
                - window_length: 입력 시퀀스 길이 (default: 100)
                - n_flows: Normalizing Flow 층 수 (default: 2)
                - use_flow: Flow 사용 여부 (default: True)
                - epochs: 최대 학습 에폭 (default: 50)
                - batch_size: 배치 크기 (default: 256)
                - learning_rate: 학습률 (default: 1e-3)
                - beta: KL 가중치 (default: 1.0)
                - device: 'cuda' or 'cpu' (default: auto)
                - early_stopping: Early Stopping 활성화 (default: True)
                - patience: Early Stopping patience (default: 5)
                - val_ratio: Validation 비율 (default: 0.1)
        """
        super().__init__()
        self.config = config

        # 기본 설정 (논문 원본 값)
        self.x_dim = config.get('x_dim', None)  # fit에서 자동 설정
        self.z_dim = config.get('z_dim', 3)           # 논문: 3
        self.hidden_dim = config.get('hidden_dim', 500)  # 논문: 500
        self.window_length = config.get('window_length', 100)  # 논문: 100
        self.n_flows = config.get('n_flows', 20)      # 논문: 20 (Planar NF)
        self.use_flow = config.get('use_flow', True)

        # 학습 설정 (논문 원본 값)
        self.epochs = config.get('epochs', 20)        # 논문: 20
        self.batch_size = config.get('batch_size', 50)  # 논문: 50
        self.learning_rate = config.get('learning_rate', 1e-3)  # 논문: 1e-3
        self.beta = config.get('beta', 1.0)
        self.n_samples = config.get('n_samples', 1)  # 이상 점수 계산 시 샘플 수
        self.weight_decay = config.get('weight_decay', 1e-4)  # 논문: L2 reg 1e-4

        # Early Stopping 설정 (원본 OmniAnomaly에도 early_stop=True 옵션 있음)
        self.early_stopping = config.get('early_stopping', True)
        self.patience = config.get('patience', 5)
        self.val_ratio = config.get('val_ratio', 0.3)  # 논문: 30%

        # Device 설정
        if config.get('device'):
            self.device = torch.device(config['device'])
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: Optional[OmniAnomalyCore] = None
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[dict] = None

    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """
        슬라이딩 윈도우로 시퀀스 생성

        Args:
            data: (n_samples, n_features)

        Returns:
            windows: (n_windows, window_length, n_features)
        """
        n_samples = len(data)
        n_windows = n_samples - self.window_length + 1

        if n_windows <= 0:
            raise ValueError(
                f"Data length ({n_samples}) is shorter than window_length ({self.window_length})"
            )

        windows = np.array([
            data[i:i + self.window_length]
            for i in range(n_windows)
        ])

        return windows

    def fit(self, train_data: np.ndarray, verbose: bool = True) -> 'OmniAnomaly':
        """
        모델 학습 (Early Stopping 지원)

        Args:
            train_data: (n_samples, n_features) 학습 데이터
            verbose: 학습 로그 출력 여부

        Returns:
            self
        """
        # x_dim 자동 설정
        if self.x_dim is None:
            self.x_dim = train_data.shape[1]

        # 모델 초기화
        self.model = OmniAnomalyCore(
            x_dim=self.x_dim,
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            n_flows=self.n_flows,
            use_flow=self.use_flow
        ).to(self.device)

        # 슬라이딩 윈도우 생성
        windows = self._create_windows(train_data)

        # Train/Validation 분할 (Early Stopping용)
        if self.early_stopping and self.val_ratio > 0:
            n_val = int(len(windows) * self.val_ratio)
            train_windows = windows[:-n_val]
            val_windows = windows[-n_val:]

            train_tensor = torch.FloatTensor(train_windows).to(self.device)
            val_tensor = torch.FloatTensor(val_windows).to(self.device)
        else:
            train_tensor = torch.FloatTensor(windows).to(self.device)
            val_tensor = None

        # DataLoader
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Optimizer (논문: Adam + L2 regularization)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay  # L2 정규화 (논문: 1e-4)
        )

        # Early Stopping 변수
        self.best_val_loss = float('inf')
        self.best_model_state = None
        patience_counter = 0

        # 학습 루프
        history = {'loss': [], 'recon_loss': [], 'kl_loss': [], 'val_loss': []}

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}', leave=False, disable=not verbose)
            for batch in pbar:
                x = batch[0]

                optimizer.zero_grad()

                # Forward
                outputs = self.model(x)

                # Loss
                losses = self.model.compute_loss(x, outputs, beta=self.beta)
                loss = losses['loss']

                # Backward
                loss.backward()

                # Gradient Clipping (수치 안정성)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                optimizer.step()

                # 기록
                epoch_loss += losses['loss'].item()
                epoch_recon += losses['recon_loss'].item()
                epoch_kl += losses['kl_loss'].item()
                n_batches += 1

                pbar.set_postfix({
                    'loss': f"{losses['loss'].item():.4f}",
                    'recon': f"{losses['recon_loss'].item():.4f}",
                    'kl': f"{losses['kl_loss'].item():.4f}"
                })

            # Epoch 평균
            avg_train_loss = epoch_loss / n_batches
            history['loss'].append(avg_train_loss)
            history['recon_loss'].append(epoch_recon / n_batches)
            history['kl_loss'].append(epoch_kl / n_batches)

            # Validation (Early Stopping)
            if self.early_stopping and val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    # Validation loss 계산 (배치로 분할)
                    val_loss_total = 0
                    val_batches = 0
                    for i in range(0, len(val_tensor), self.batch_size):
                        val_batch = val_tensor[i:i + self.batch_size]
                        if len(val_batch) < 2:  # 너무 작은 배치 스킵
                            continue
                        val_outputs = self.model(val_batch)
                        val_losses = self.model.compute_loss(val_batch, val_outputs, beta=self.beta)
                        val_loss_total += val_losses['loss'].item()
                        val_batches += 1

                    val_loss = val_loss_total / max(val_batches, 1)
                    history['val_loss'].append(val_loss)

                # Best model 저장
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={val_loss:.4f}")

                # Early Stopping 체크
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  ⚡ Early stopping at epoch {epoch+1} (patience={self.patience})")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, "
                          f"Recon={history['recon_loss'][-1]:.4f}, KL={history['kl_loss'][-1]:.4f}")

        # Best model 복원
        if self.early_stopping and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"  ✓ Best model restored (val_loss: {self.best_val_loss:.4f})")

        self.is_fitted = True
        self.history = history
        return self

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        복원된 데이터 반환

        Args:
            test_data: (n_samples, n_features)

        Returns:
            reconstructed: (n_samples, n_features)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        # 윈도우 생성
        windows = self._create_windows(test_data)
        windows_tensor = torch.FloatTensor(windows).to(self.device)

        reconstructed_list = []

        with torch.no_grad():
            for i in range(0, len(windows_tensor), self.batch_size):
                batch = windows_tensor[i:i + self.batch_size]
                outputs = self.model(batch)

                # 마지막 시점의 복원값 사용
                x_mu = outputs['x_mu'][:, -1, :]  # (batch, x_dim)
                reconstructed_list.append(x_mu.cpu().numpy())

        reconstructed = np.concatenate(reconstructed_list, axis=0)

        # 첫 window_length-1 개는 패딩 (첫 번째 복원값으로)
        padding = np.tile(reconstructed[0], (self.window_length - 1, 1))
        reconstructed = np.vstack([padding, reconstructed])

        return reconstructed

    def get_anomaly_score(self, test_data: np.ndarray) -> np.ndarray:
        """
        이상 점수 계산

        이상 점수 = -복원 확률 (Negative Log Probability)
        클수록 이상일 가능성이 높음

        Args:
            test_data: (n_samples, n_features)

        Returns:
            scores: (n_samples,) 각 시점의 이상 점수
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()

        # 윈도우 생성
        windows = self._create_windows(test_data)
        windows_tensor = torch.FloatTensor(windows).to(self.device)

        scores_list = []

        with torch.no_grad():
            for i in range(0, len(windows_tensor), self.batch_size):
                batch = windows_tensor[i:i + self.batch_size]

                # 이상 점수 (마지막 시점)
                scores = self.model.get_anomaly_score(
                    batch,
                    n_samples=self.n_samples,
                    last_point_only=True
                )
                scores_list.append(scores.cpu().numpy())

        scores = np.concatenate(scores_list, axis=0)

        # 첫 window_length-1 개는 패딩 (첫 번째 점수로)
        padding = np.full(self.window_length - 1, scores[0])
        scores = np.concatenate([padding, scores])

        return scores

    def save(self, path: str) -> None:
        """모델 저장"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Nothing to save.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'x_dim': self.x_dim
        }, path)

    def load(self, path: str) -> 'OmniAnomaly':
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)

        self.config = checkpoint['config']
        self.x_dim = checkpoint['x_dim']

        # 모델 재생성
        self.model = OmniAnomalyCore(
            x_dim=self.x_dim,
            z_dim=self.z_dim,
            hidden_dim=self.hidden_dim,
            n_flows=self.n_flows,
            use_flow=self.use_flow
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True

        return self
