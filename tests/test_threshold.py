# -*- coding: utf-8 -*-
"""
Threshold module test script
Usage: python tests/test_threshold.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from postprocessing import FixedThreshold, EWMAThreshold, AdaptiveThreshold


def test_threshold_implementation():
    """Threshold module implementation test"""

    print("=" * 60)
    print("🧪 Threshold Module Test")
    print("=" * 60)

    all_passed = True
    np.random.seed(42)

    # Create test anomaly scores
    # Normal scores (low) + some anomaly scores (high)
    n_normal = 90
    n_anomaly = 10

    normal_scores = np.random.normal(loc=0.2, scale=0.1, size=n_normal)
    anomaly_scores = np.random.normal(loc=0.8, scale=0.1, size=n_anomaly)

    # Combine and create labels
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    y_true = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    # Shuffle
    indices = np.random.permutation(len(all_scores))
    all_scores = all_scores[indices]
    y_true = y_true[indices]

    print(f"\n📊 Test Data Info:")
    print(f"  - Total samples: {len(all_scores)}")
    print(f"  - Normal: {n_normal}, Anomaly: {n_anomaly}")
    print(f"  - Score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")

    # ==================== Test 1: FixedThreshold (Sigma) ====================
    print("\n[Test 1] FixedThreshold (Sigma method)")
    print("-" * 40)

    try:
        threshold_sigma = FixedThreshold(method='sigma', n_sigma=2.0)
        threshold_sigma.fit(all_scores)

        print(f"✓ Mean: {threshold_sigma.mean_:.4f}")
        print(f"✓ Std: {threshold_sigma.std_:.4f}")
        print(f"✓ Threshold (mean + 2*std): {threshold_sigma.threshold_:.4f}")

        predictions = threshold_sigma.apply(all_scores)
        n_detected = np.sum(predictions)
        print(f"✓ Detected anomalies: {n_detected}")

        # Check threshold formula
        expected_threshold = threshold_sigma.mean_ + 2.0 * threshold_sigma.std_
        if abs(threshold_sigma.threshold_ - expected_threshold) < 1e-6:
            print("✅ Sigma threshold formula correct!")
        else:
            print("❌ Sigma threshold formula error")
            all_passed = False

    except NotImplementedError as e:
        print(f"❌ TODO(human) Implementation needed: {e}")
        print("   → src/postprocessing/threshold.py의 FixedThreshold.fit() sigma 부분을 구현하세요")
        all_passed = False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 2: FixedThreshold (Percentile) ====================
    print("\n[Test 2] FixedThreshold (Percentile method)")
    print("-" * 40)

    try:
        threshold_pct = FixedThreshold(method='percentile', percentile=90)
        threshold_pct.fit(all_scores)

        print(f"✓ Threshold (90th percentile): {threshold_pct.threshold_:.4f}")

        predictions_pct = threshold_pct.apply(all_scores)
        n_detected_pct = np.sum(predictions_pct)
        print(f"✓ Detected anomalies: {n_detected_pct}")

        # 90th percentile should detect ~10% as anomalies
        detection_rate = n_detected_pct / len(all_scores) * 100
        print(f"✓ Detection rate: {detection_rate:.1f}%")

        if 5 <= detection_rate <= 15:
            print("✅ Percentile threshold works correctly!")
        else:
            print(f"⚠️ Detection rate should be around 10%")

    except Exception as e:
        print(f"❌ Error: {e}")
        all_passed = False

    # ==================== Test 3: EWMAThreshold ====================
    print("\n[Test 3] EWMAThreshold (Dynamic)")
    print("-" * 40)

    try:
        threshold_ewma = EWMAThreshold(span=20, n_sigma=2.0)
        threshold_ewma.fit(all_scores)

        print(f"✓ Dynamic threshold shape: {threshold_ewma.threshold_.shape}")
        print(f"✓ Threshold range: [{threshold_ewma.threshold_.min():.4f}, {threshold_ewma.threshold_.max():.4f}]")

        predictions_ewma = threshold_ewma.apply(all_scores)
        n_detected_ewma = np.sum(predictions_ewma)
        print(f"✓ Detected anomalies: {n_detected_ewma}")

        print("✅ EWMA threshold works!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 4: AdaptiveThreshold ====================
    print("\n[Test 4] AdaptiveThreshold (Best F1)")
    print("-" * 40)

    try:
        threshold_adaptive = AdaptiveThreshold(n_thresholds=100)
        threshold_adaptive.fit(all_scores, y_true=y_true)

        print(f"✓ Best threshold: {threshold_adaptive.threshold_:.4f}")
        print(f"✓ Best F1-score: {threshold_adaptive.best_f1_:.4f}")

        predictions_adaptive = threshold_adaptive.apply(all_scores)

        # Calculate metrics
        tp = np.sum((predictions_adaptive == 1) & (y_true == 1))
        fp = np.sum((predictions_adaptive == 1) & (y_true == 0))
        fn = np.sum((predictions_adaptive == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"✓ TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"✓ Precision: {precision:.4f}, Recall: {recall:.4f}")

        if threshold_adaptive.best_f1_ > 0.7:
            print("✅ Adaptive threshold achieves good F1!")
        else:
            print("⚠️ F1 could be higher with better separation")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 5: Comparison ====================
    print("\n[Test 5] Threshold Methods Comparison")
    print("-" * 40)

    try:
        methods = {
            'Sigma (2σ)': FixedThreshold(method='sigma', n_sigma=2.0),
            'Sigma (3σ)': FixedThreshold(method='sigma', n_sigma=3.0),
            'Percentile (90%)': FixedThreshold(method='percentile', percentile=90),
            'Percentile (95%)': FixedThreshold(method='percentile', percentile=95),
        }

        print(f"{'Method':<20} {'Threshold':>10} {'Detected':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-" * 60)

        for name, method in methods.items():
            method.fit(all_scores)
            preds = method.apply(all_scores)

            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            print(f"{name:<20} {method.threshold_:>10.4f} {np.sum(preds):>10} {tp:>5} {fp:>5} {fn:>5}")

        print("✅ Comparison complete!")

    except NotImplementedError:
        print("⚠️ Sigma method not implemented yet")
    except Exception as e:
        print(f"❌ Error: {e}")
        all_passed = False

    # ==================== Result ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Threshold module complete.")
    else:
        print("⚠️ Some tests failed. Check errors above.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_threshold_implementation()
