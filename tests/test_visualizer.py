# -*- coding: utf-8 -*-
"""
AnomalyVisualizer test script
Usage: python tests/test_visualizer.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from evaluation.visualizer import AnomalyVisualizer, quick_plot_scores, quick_plot_decision


def test_visualizer():
    """AnomalyVisualizer implementation test"""

    print("=" * 60)
    print("Visualizer Test")
    print("=" * 60)

    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 500

    # 정상 Score (낮은 값)
    normal_scores = np.random.exponential(0.3, n_samples)

    # 이상 구간 설정 (3개 구간)
    labels = np.zeros(n_samples)
    labels[100:130] = 1  # 첫 번째 이상 구간
    labels[250:280] = 1  # 두 번째 이상 구간
    labels[400:420] = 1  # 세 번째 이상 구간

    # 이상 구간에서 Score 증가
    scores = normal_scores.copy()
    scores[100:130] += np.random.uniform(1.0, 2.0, 30)
    scores[250:280] += np.random.uniform(0.8, 1.5, 30)
    scores[400:420] += np.random.uniform(1.2, 2.5, 20)

    # Threshold 설정
    threshold = np.percentile(scores, 95)

    # Binary Prediction 생성
    predictions = (scores > threshold).astype(int)

    all_passed = True

    try:
        # ==================== Test 1: 초기화 ====================
        print("\n[Test 1] Initialization")
        print("-" * 40)

        visualizer = AnomalyVisualizer(figsize=(12, 5), dpi=80)

        assert visualizer.figsize == (12, 5)
        assert visualizer.dpi == 80
        assert 'score' in visualizer.colors
        assert 'tp' in visualizer.colors

        print("Visualizer initialized successfully!")

        # ==================== Test 2: Anomaly Score Plot ====================
        print("\n[Test 2] Anomaly Score Plot")
        print("-" * 40)

        fig = visualizer.plot_anomaly_scores(
            scores=scores,
            labels=labels,
            threshold=threshold,
            title='Test: Anomaly Scores',
            highlight_anomalies=True
        )

        assert fig is not None
        assert len(fig.axes) == 1

        # Figure 저장 테스트
        test_output_dir = Path('/tmp/test_visualizer')
        test_output_dir.mkdir(exist_ok=True)

        save_path = visualizer.save_figure(fig, test_output_dir / 'score_plot.png')
        assert Path(save_path).exists()

        plt_close(fig)
        print("Score plot created and saved!")

        # ==================== Test 3: Binary Decision Plot ====================
        print("\n[Test 3] Binary Decision Plot")
        print("-" * 40)

        fig = visualizer.plot_binary_decision(
            predictions=predictions,
            labels=labels,
            title='Test: Binary Decision',
            show_metrics=True
        )

        assert fig is not None
        assert len(fig.axes) == 2  # 2개 subplot

        save_path = visualizer.save_figure(fig, test_output_dir / 'decision_plot.png')
        assert Path(save_path).exists()

        plt_close(fig)
        print("Decision plot created!")
        print("Note: TODO(human) for TP/FP/FN masks needs implementation")

        # ==================== Test 4: Comparison Plot ====================
        print("\n[Test 4] Comparison Bar Chart")
        print("-" * 40)

        results = [
            {'label': 'DLinear+T1', 'pa_f1': 0.72},
            {'label': 'DLinear+T2', 'pa_f1': 0.78},
            {'label': 'DLinear+T3', 'pa_f1': 0.85},
            {'label': 'OmniAnomaly+T1', 'pa_f1': 0.70},
            {'label': 'OmniAnomaly+T3', 'pa_f1': 0.82},
        ]

        fig = visualizer.plot_comparison(
            results=results,
            metric='pa_f1',
            title='Model Comparison: PA F1 Score'
        )

        assert fig is not None

        save_path = visualizer.save_figure(fig, test_output_dir / 'comparison.png')
        assert Path(save_path).exists()

        plt_close(fig)
        print("Comparison chart created!")

        # ==================== Test 5: Score Distribution ====================
        print("\n[Test 5] Score Distribution Histogram")
        print("-" * 40)

        fig = visualizer.plot_score_distribution(
            scores=scores,
            labels=labels,
            threshold=threshold,
            title='Test: Score Distribution'
        )

        assert fig is not None

        save_path = visualizer.save_figure(fig, test_output_dir / 'distribution.png')
        assert Path(save_path).exists()

        plt_close(fig)
        print("Distribution plot created!")

        # ==================== Test 6: Quick Functions ====================
        print("\n[Test 6] Quick Plot Functions")
        print("-" * 40)

        fig1 = quick_plot_scores(scores, labels, threshold, 'Quick Score Plot')
        fig2 = quick_plot_decision(predictions, labels, 'Quick Decision Plot')

        assert fig1 is not None
        assert fig2 is not None

        plt_close(fig1)
        plt_close(fig2)
        print("Quick functions work!")

        # ==================== Test 7: Anomaly Region Detection ====================
        print("\n[Test 7] Anomaly Region Detection")
        print("-" * 40)

        regions = visualizer._find_anomaly_regions(labels)

        print(f"Detected anomaly regions: {regions}")
        assert len(regions) == 3
        assert regions[0] == (100, 130)
        assert regions[1] == (250, 280)
        assert regions[2] == (400, 420)

        print("Anomaly regions correctly identified!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Result ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! AnomalyVisualizer is ready.")
        print(f"\nTest outputs saved to: /tmp/test_visualizer/")
        print("  - score_plot.png")
        print("  - decision_plot.png")
        print("  - comparison.png")
        print("  - distribution.png")
    else:
        print("Some tests failed. Check errors above.")
    print("=" * 60)

    return all_passed


def plt_close(fig):
    """Close matplotlib figure"""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == '__main__':
    test_visualizer()
