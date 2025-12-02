# -*- coding: utf-8 -*-
"""
OmniAnomaly Core Implementation (PyTorch)

Original: https://github.com/NetManAIOps/OmniAnomaly
Paper: "Robust Anomaly Detection for Multivariate Time Series" (KDD 2019)

🎓 [학습 포인트]
OmniAnomaly는 Reconstruction-based 이상 탐지 모델로:
1. GRU 기반 인코더로 시계열 패턴 학습
2. VAE로 잠재 공간에서 정상 패턴 모델링
3. Planar Normalizing Flow로 사후 분포 표현력 향상
4. 복원 확률(reconstruction probability)로 이상 점수 계산

Architecture:
    x → GRU Encoder → q(z|x) → z → GRU Decoder → p(x|z) → x_reconstructed
                        ↓
                  Normalizing Flow (optional)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class GRUEncoder(nn.Module):
    """
    GRU 기반 인코더

    시계열 입력을 받아 잠재 변수 z의 평균과 분산을 출력합니다.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=x_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # z의 평균과 로그 분산을 출력하는 레이어
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, x_dim)

        Returns:
            mu: (batch, seq_len, z_dim) - 잠재 변수 평균
            logvar: (batch, seq_len, z_dim) - 잠재 변수 로그 분산
        """
        # GRU 인코딩
        h, _ = self.gru(x)  # (batch, seq_len, hidden_dim)

        # 평균과 로그 분산 계산
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class GRUDecoder(nn.Module):
    """
    GRU 기반 디코더

    잠재 변수 z를 받아 원본 x를 복원합니다.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        hidden_dim: int = 100,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=z_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 복원된 x의 평균과 로그 분산
        self.fc_mu = nn.Linear(hidden_dim, x_dim)
        self.fc_logvar = nn.Linear(hidden_dim, x_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, seq_len, z_dim)

        Returns:
            x_mu: (batch, seq_len, x_dim) - 복원된 x의 평균
            x_logvar: (batch, seq_len, x_dim) - 복원된 x의 로그 분산
        """
        h, _ = self.gru(z)  # (batch, seq_len, hidden_dim)

        x_mu = self.fc_mu(h)
        x_logvar = self.fc_logvar(h)

        return x_mu, x_logvar


class PlanarFlow(nn.Module):
    """
    Planar Normalizing Flow

    간단하면서도 효과적인 Normalizing Flow 구현.
    z' = z + u * tanh(w^T * z + b)

    🎓 [학습 포인트]
    - Normalizing Flow: 단순한 분포를 복잡한 분포로 변환
    - Planar Flow: 평면(hyperplane)을 따라 분포를 변형
    - 여러 층을 쌓아 표현력 향상
    """

    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim

        # Planar flow 파라미터
        self.u = nn.Parameter(torch.randn(z_dim) * 0.01)
        self.w = nn.Parameter(torch.randn(z_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, ..., z_dim)

        Returns:
            z': 변환된 z
            log_det: 로그 야코비안 행렬식
        """
        # w^T * z + b
        linear = F.linear(z, self.w.unsqueeze(0), self.b)  # (..., 1)
        linear = linear.squeeze(-1)  # (...)

        # tanh activation
        tanh_linear = torch.tanh(linear)

        # z' = z + u * tanh(w^T * z + b)
        z_new = z + self.u * tanh_linear.unsqueeze(-1)

        # 로그 야코비안: log|1 + u^T * w * (1 - tanh^2)|
        psi = (1 - tanh_linear ** 2).unsqueeze(-1) * self.w  # (..., z_dim)
        det = 1 + (psi * self.u).sum(dim=-1)
        log_det = torch.log(torch.abs(det) + 1e-8)

        return z_new, log_det


class NormalizingFlows(nn.Module):
    """여러 Planar Flow 층을 쌓은 모듈"""

    def __init__(self, z_dim: int, n_flows: int = 2):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(z_dim) for _ in range(n_flows)])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_k: 최종 변환된 z
            sum_log_det: 누적 로그 야코비안
        """
        sum_log_det = 0
        z_k = z

        for flow in self.flows:
            z_k, log_det = flow(z_k)
            sum_log_det = sum_log_det + log_det

        return z_k, sum_log_det


class OmniAnomalyCore(nn.Module):
    """
    OmniAnomaly 핵심 모델 (PyTorch 버전)

    구성 요소:
    1. GRU Encoder: x → (mu, logvar)
    2. Reparameterization: z = mu + std * epsilon
    3. (Optional) Normalizing Flow: z → z'
    4. GRU Decoder: z' → (x_mu, x_logvar)

    손실 함수: ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

    🎓 [학습 포인트]
    - VAE의 핵심: 잠재 공간에서 정상 패턴 학습
    - 이상 데이터는 잠재 공간에서 잘 표현되지 않음 → 복원 오류 증가
    - Normalizing Flow로 사후 분포의 표현력 향상
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int = 8,
        hidden_dim: int = 100,
        num_layers: int = 1,
        n_flows: int = 2,
        use_flow: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            x_dim: 입력 차원 (feature 수)
            z_dim: 잠재 공간 차원
            hidden_dim: GRU hidden 차원
            num_layers: GRU 층 수
            n_flows: Normalizing Flow 층 수
            use_flow: Normalizing Flow 사용 여부
            dropout: Dropout 비율
        """
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.use_flow = use_flow

        # Encoder & Decoder
        self.encoder = GRUEncoder(x_dim, z_dim, hidden_dim, num_layers, dropout)
        self.decoder = GRUDecoder(x_dim, z_dim, hidden_dim, num_layers, dropout)

        # Normalizing Flow (optional)
        if use_flow and n_flows > 0:
            self.flow = NormalizingFlows(z_dim, n_flows)
        else:
            self.flow = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick

        z = mu + std * epsilon, where epsilon ~ N(0, I)

        🎓 [학습 포인트]
        - VAE에서 gradient가 sampling을 통과할 수 있도록 하는 핵심 기법
        - mu, logvar는 학습 가능, epsilon은 노이즈
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        인코딩: x → z

        Args:
            x: (batch, seq_len, x_dim)

        Returns:
            z: 샘플링된 잠재 변수
            mu: 잠재 변수 평균
            logvar: 잠재 변수 로그 분산
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        디코딩: z → x_reconstructed

        Args:
            z: (batch, seq_len, z_dim)

        Returns:
            x_mu: 복원된 x의 평균
            x_logvar: 복원된 x의 로그 분산
        """
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        순전파

        Args:
            x: (batch, seq_len, x_dim)

        Returns:
            dict with:
                - x_mu: 복원된 x의 평균
                - x_logvar: 복원된 x의 로그 분산
                - z: 잠재 변수
                - z_mu: 잠재 변수 평균
                - z_logvar: 잠재 변수 로그 분산
                - flow_log_det: Normalizing Flow 로그 야코비안 (사용 시)
        """
        # Encode
        z, z_mu, z_logvar = self.encode(x)

        # Apply Normalizing Flow (optional)
        flow_log_det = None
        if self.flow is not None:
            z, flow_log_det = self.flow(z)

        # Decode
        x_mu, x_logvar = self.decode(z)

        return {
            'x_mu': x_mu,
            'x_logvar': x_logvar,
            'z': z,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'flow_log_det': flow_log_det
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        beta: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        ELBO 손실 계산

        Loss = -E[log p(x|z)] + beta * KL(q(z|x) || p(z))

        Args:
            x: 원본 입력
            outputs: forward() 출력
            beta: KL divergence 가중치 (beta-VAE)

        Returns:
            dict with loss components
        """
        x_mu = outputs['x_mu']
        x_logvar = outputs['x_logvar']
        z_mu = outputs['z_mu']
        z_logvar = outputs['z_logvar']
        flow_log_det = outputs.get('flow_log_det')

        # Reconstruction Loss: Negative log-likelihood under Gaussian
        # -log p(x|z) = 0.5 * (log(2π) + logvar + (x - mu)^2 / var)
        # 상수항 log(2π)는 최적화에 영향 없으므로 생략 가능
        x_std = torch.exp(0.5 * x_logvar)
        dist = Normal(x_mu, x_std)
        log_prob = dist.log_prob(x)  # (batch, seq_len, x_dim)
        recon_loss = -log_prob.sum(dim=-1).mean()  # negative log prob, sum over features

        # KL Divergence: KL(q(z|x) || p(z)), where p(z) = N(0, I)
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * (1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        kl_loss = kl_loss.sum(dim=-1).mean()

        # Normalizing Flow 보정 (사용 시)
        if flow_log_det is not None:
            kl_loss = kl_loss - flow_log_det.mean()

        # Total ELBO loss
        total_loss = recon_loss + beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def get_reconstruction_prob(
        self,
        x: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        복원 확률 계산 (이상 점수의 기반)

        여러 번 샘플링하여 평균 복원 확률을 계산합니다.
        점수가 낮을수록 이상일 가능성이 높습니다.

        Args:
            x: (batch, seq_len, x_dim)
            n_samples: z 샘플 수

        Returns:
            recon_prob: (batch, seq_len) - 각 시점의 복원 확률
        """
        total_log_prob = 0

        for _ in range(n_samples):
            outputs = self.forward(x)
            x_mu = outputs['x_mu']
            x_logvar = outputs['x_logvar']

            # Log probability under Gaussian
            x_std = torch.exp(0.5 * x_logvar)
            dist = Normal(x_mu, x_std)
            log_prob = dist.log_prob(x)  # (batch, seq_len, x_dim)
            log_prob = log_prob.sum(dim=-1)  # (batch, seq_len)

            total_log_prob = total_log_prob + log_prob

        # 평균
        mean_log_prob = total_log_prob / n_samples

        return mean_log_prob

    def get_anomaly_score(
        self,
        x: torch.Tensor,
        n_samples: int = 1,
        last_point_only: bool = True
    ) -> torch.Tensor:
        """
        이상 점수 계산

        이상 점수 = -복원 확률 (클수록 이상)

        Args:
            x: (batch, seq_len, x_dim)
            n_samples: z 샘플 수
            last_point_only: 마지막 시점만 반환할지 여부

        Returns:
            scores: 이상 점수 (클수록 이상)
        """
        recon_prob = self.get_reconstruction_prob(x, n_samples)

        # 음의 복원 확률 = 이상 점수
        scores = -recon_prob

        if last_point_only:
            scores = scores[:, -1]  # (batch,)

        return scores
