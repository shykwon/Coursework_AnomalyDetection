# -*- coding: utf-8 -*-
"""
이상 탐지 시각화 모듈

🎓 [학습 포인트]
- Anomaly Score Plot: 시계열 Score + Threshold + 실제 이상 구간
- Binary Decision Plot: TP/FP/FN 색상 구분
- 보고서용 고품질 Figure 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from pathlib import Path


class AnomalyVisualizer:
    """
    이상 탐지 결과 시각화 클래스

    사용법:
    ```python
    visualizer = AnomalyVisualizer(figsize=(14, 6), dpi=100)

    # Anomaly Score Plot
    fig = visualizer.plot_anomaly_scores(
        scores=anomaly_scores,
        labels=ground_truth,
        threshold=0.5,
        title='DLinear + P_MM: Anomaly Scores'
    )
    fig.savefig('anomaly_scores.png')

    # Binary Decision Plot
    fig = visualizer.plot_binary_decision(
        predictions=binary_preds,
        labels=ground_truth,
        title='DLinear + P_MM + T1: Detection Results'
    )
    ```
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 6),
        dpi: int = 100,
        style: str = 'seaborn-v0_8-whitegrid'
    ):
        """
        Args:
            figsize: Figure 크기 (width, height)
            dpi: Figure 해상도
            style: Matplotlib 스타일
        """
        self.figsize = figsize
        self.dpi = dpi
        self.style = style

        # 색상 팔레트
        self.colors = {
            'score': '#2196F3',       # Blue - Anomaly Score
            'threshold': '#F44336',    # Red - Threshold line
            'anomaly_region': '#FFCDD2',  # Light Red - 실제 이상 구간
            'tp': '#4CAF50',           # Green - True Positive
            'fp': '#FF9800',           # Orange - False Positive
            'fn': '#F44336',           # Red - False Negative
            'tn': '#E0E0E0',           # Light Gray - True Negative
            'prediction': '#9C27B0',   # Purple - Prediction line
        }

    def plot_anomaly_scores(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        title: str = 'Anomaly Scores',
        xlabel: str = 'Time',
        ylabel: str = 'Anomaly Score',
        show_legend: bool = True,
        highlight_anomalies: bool = True
    ) -> plt.Figure:
        """
        Anomaly Score 시계열 플롯

        Args:
            scores: Anomaly Score 배열
            labels: Ground Truth 라벨 (0: 정상, 1: 이상)
            threshold: 탐지 임계값 (선 표시)
            title: 그래프 제목
            xlabel: X축 라벨
            ylabel: Y축 라벨
            show_legend: 범례 표시 여부
            highlight_anomalies: 실제 이상 구간 하이라이트

        Returns:
            matplotlib Figure 객체
        """
        try:
            plt.style.use(self.style)
        except OSError:
            pass  # 스타일 없으면 기본 스타일 사용

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        time_idx = np.arange(len(scores))

        # 실제 이상 구간 하이라이트
        if labels is not None and highlight_anomalies:
            anomaly_regions = self._find_anomaly_regions(labels)
            for start, end in anomaly_regions:
                ax.axvspan(
                    start, end,
                    alpha=0.3,
                    color=self.colors['anomaly_region'],
                    label='Actual Anomaly' if start == anomaly_regions[0][0] else ''
                )

        # Anomaly Score 플롯
        ax.plot(
            time_idx, scores,
            color=self.colors['score'],
            linewidth=0.8,
            alpha=0.8,
            label='Anomaly Score'
        )

        # Threshold 라인
        if threshold is not None:
            ax.axhline(
                y=threshold,
                color=self.colors['threshold'],
                linestyle='--',
                linewidth=1.5,
                label=f'Threshold ({threshold:.3f})'
            )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        if show_legend:
            ax.legend(loc='upper right', fontsize=9)

        ax.set_xlim(0, len(scores))

        plt.tight_layout()
        return fig

    def plot_binary_decision(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        title: str = 'Binary Decision Results',
        xlabel: str = 'Time',
        show_legend: bool = True,
        show_metrics: bool = True
    ) -> plt.Figure:
        """
        Binary Decision 결과 시각화 (TP/FP/FN 색상 구분)

        Args:
            predictions: 예측 라벨 (0: 정상, 1: 이상)
            labels: Ground Truth 라벨 (0: 정상, 1: 이상)
            title: 그래프 제목
            xlabel: X축 라벨
            show_legend: 범례 표시 여부
            show_metrics: 성능 지표 텍스트 표시

        Returns:
            matplotlib Figure 객체
        """
        try:
            plt.style.use(self.style)
        except OSError:
            pass

        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2),
                                  dpi=self.dpi, sharex=True)

        time_idx = np.arange(len(predictions))

        # 상단: Ground Truth
        ax1 = axes[0]
        ax1.fill_between(
            time_idx, 0, labels,
            color=self.colors['anomaly_region'],
            alpha=0.7,
            label='Ground Truth'
        )
        ax1.set_ylabel('Ground Truth', fontsize=10)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Normal', 'Anomaly'])
        if show_legend:
            ax1.legend(loc='upper right', fontsize=9)

        # 하단: Prediction with TP/FP/FN coloring
        ax2 = axes[1]

        # ============================================================
        # TODO(human): TP/FP/FN 분류 및 색상 지정
        # ============================================================
        # 이진 분류에서 각 포인트를 TP, FP, FN, TN으로 분류합니다.
        # - TP (True Positive): 실제 이상(1)을 이상(1)으로 예측
        # - FP (False Positive): 실제 정상(0)을 이상(1)으로 예측
        # - FN (False Negative): 실제 이상(1)을 정상(0)으로 예측
        # - TN (True Negative): 실제 정상(0)을 정상(0)으로 예측
        #
        # Hint: numpy의 논리 연산자 (&, |) 사용


        tp_mask = (predictions == 1) & (labels == 1)
        fp_mask = (predictions == 1) & (labels == 0)
        fn_mask = (predictions == 0) & (labels == 1)

        # TP, FP, FN 영역 색상 표시
        # TP: 초록색
        ax2.fill_between(
            time_idx, 0, predictions * tp_mask,
            color=self.colors['tp'],
            alpha=0.7,
            label='TP (Correct Detection)'
        )
        # FP: 주황색
        ax2.fill_between(
            time_idx, 0, predictions * fp_mask,
            color=self.colors['fp'],
            alpha=0.7,
            label='FP (False Alarm)'
        )
        # FN: 빨간색 (예측은 0이지만 실제는 1인 구간 표시)
        ax2.fill_between(
            time_idx, 0, fn_mask.astype(float),
            color=self.colors['fn'],
            alpha=0.4,
            label='FN (Missed)'
        )

        ax2.set_ylabel('Prediction', fontsize=10)
        ax2.set_xlabel(xlabel, fontsize=11)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Normal', 'Anomaly'])

        if show_legend:
            ax2.legend(loc='upper right', fontsize=9)

        # 성능 지표 표시
        if show_metrics:
            tp = np.sum(tp_mask)
            fp = np.sum(fp_mask)
            fn = np.sum(fn_mask)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics_text = f'P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}'
            ax2.text(
                0.02, 0.95, metrics_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig

    def plot_comparison(
        self,
        results: List[Dict],
        metric: str = 'pa_f1',
        title: str = 'Model Comparison',
        xlabel: str = 'Configuration',
        ylabel: str = 'PA F1 Score'
    ) -> plt.Figure:
        """
        여러 실험 결과 비교 바 차트

        Args:
            results: 실험 결과 리스트
                [{'label': 'DLinear+P_MM+T1', 'pa_f1': 0.85, ...}, ...]
            metric: 비교할 지표 키
            title: 그래프 제목
            xlabel: X축 라벨
            ylabel: Y축 라벨

        Returns:
            matplotlib Figure 객체
        """
        try:
            plt.style.use(self.style)
        except OSError:
            pass

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        labels = [r.get('label', f"Exp {i+1}") for i, r in enumerate(results)]
        values = [r.get(metric, 0) for r in results]

        # 바 색상 (최고 성능은 다른 색)
        colors = [self.colors['score']] * len(values)
        if values:
            max_idx = np.argmax(values)
            colors[max_idx] = self.colors['tp']

        bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')

        # 값 라벨 표시
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')

        # X축 라벨 회전
        plt.xticks(rotation=45, ha='right')

        ax.set_ylim(0, max(values) * 1.15 if values else 1)

        plt.tight_layout()
        return fig

    def plot_score_distribution(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
        title: str = 'Anomaly Score Distribution',
        bins: int = 50
    ) -> plt.Figure:
        """
        Anomaly Score 분포 히스토그램

        Args:
            scores: Anomaly Score 배열
            labels: Ground Truth 라벨 (분리 히스토그램용)
            threshold: 탐지 임계값
            title: 그래프 제목
            bins: 히스토그램 빈 개수

        Returns:
            matplotlib Figure 객체
        """
        try:
            plt.style.use(self.style)
        except OSError:
            pass

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if labels is not None:
            # 정상/이상 분리 히스토그램
            normal_scores = scores[labels == 0]
            anomaly_scores = scores[labels == 1]

            ax.hist(
                normal_scores, bins=bins, alpha=0.6,
                color=self.colors['tp'], label='Normal',
                density=True
            )
            ax.hist(
                anomaly_scores, bins=bins, alpha=0.6,
                color=self.colors['fn'], label='Anomaly',
                density=True
            )
        else:
            ax.hist(
                scores, bins=bins, alpha=0.7,
                color=self.colors['score'],
                density=True
            )

        if threshold is not None:
            ax.axvline(
                x=threshold,
                color=self.colors['threshold'],
                linestyle='--',
                linewidth=2,
                label=f'Threshold ({threshold:.3f})'
            )

        ax.set_xlabel('Anomaly Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        return fig

    def _find_anomaly_regions(self, labels: np.ndarray) -> List[Tuple[int, int]]:
        """
        연속된 이상 구간 찾기

        Args:
            labels: Ground Truth 라벨 배열

        Returns:
            (start, end) 튜플 리스트
        """
        regions = []
        in_anomaly = False
        start = 0

        for i, label in enumerate(labels):
            if label == 1 and not in_anomaly:
                start = i
                in_anomaly = True
            elif label == 0 and in_anomaly:
                regions.append((start, i))
                in_anomaly = False

        # 마지막 구간 처리
        if in_anomaly:
            regions.append((start, len(labels)))

        return regions

    def save_figure(
        self,
        fig: plt.Figure,
        filepath: str,
        format: str = 'png',
        bbox_inches: str = 'tight'
    ) -> str:
        """
        Figure 저장

        Args:
            fig: matplotlib Figure 객체
            filepath: 저장 경로
            format: 이미지 포맷 ('png', 'pdf', 'svg')
            bbox_inches: 경계 설정

        Returns:
            저장된 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filepath, format=format, bbox_inches=bbox_inches, dpi=self.dpi)
        print(f"✅ Figure saved: {filepath}")

        return str(filepath)


# 편의 함수
def quick_plot_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    title: str = 'Anomaly Scores'
) -> plt.Figure:
    """빠른 Score 플롯 (편의 함수)"""
    visualizer = AnomalyVisualizer()
    return visualizer.plot_anomaly_scores(scores, labels, threshold, title)


def quick_plot_decision(
    predictions: np.ndarray,
    labels: np.ndarray,
    title: str = 'Detection Results'
) -> plt.Figure:
    """빠른 Decision 플롯 (편의 함수)"""
    visualizer = AnomalyVisualizer()
    return visualizer.plot_binary_decision(predictions, labels, title)
