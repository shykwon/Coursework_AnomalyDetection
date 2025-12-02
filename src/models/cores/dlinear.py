# -*- coding: utf-8 -*-
"""
DLinear 모델 코어
원본: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py

⚠️ 주의: 이 파일은 오픈소스 코드를 기반으로 합니다.
핵심 알고리즘 로직은 원본과 100% 동일하게 유지해야 합니다.
변경 사항: Config 클래스 추가 (argparse → dataclass 어댑터)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ============================================================
# Config 어댑터 (원본은 argparse 사용, 여기서는 dataclass로 변환)
# ============================================================
@dataclass
class DLinearConfig:
    """
    DLinear 설정 클래스

    🎓 [학습 포인트]
    원본 DLinear는 argparse로 설정을 받지만,
    우리 프레임워크에서는 dataclass로 관리합니다.

    Attributes:
        seq_len: 입력 시퀀스 길이 (lookback window)
        pred_len: 예측 시퀀스 길이 (이상치 탐지에서는 보통 1)
        enc_in: 입력 채널 수 (feature 수)
        individual: True면 채널별 독립 Linear, False면 공유
    """
    seq_len: int = 100      # 입력 윈도우 크기
    pred_len: int = 1       # 예측 길이 (이상치 탐지: 다음 1 step 예측)
    enc_in: int = 25        # feature 수 (데이터셋에 따라 조정)
    individual: bool = False  # 채널 독립 여부


# ============================================================
# 아래는 원본 DLinear 코드 (수정 없음)
# 출처: https://github.com/cure-lab/LTSF-Linear
# ============================================================

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    Decomposition-Linear (DLinear) 모델

    🎓 [학습 포인트]
    1. Series Decomposition: Moving Avg로 Trend 추출, 나머지가 Seasonal
    2. Linear Layer: 각 성분에 독립적인 Linear 적용
    3. 예측 = Seasonal 예측 + Trend 예측

    Input shape: [Batch, seq_len, n_features]
    Output shape: [Batch, pred_len, n_features]
    """
    def __init__(self, configs: DLinearConfig):
        super(DLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype
            ).to(trend_init.device)

            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
