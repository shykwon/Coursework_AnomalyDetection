# -*- coding: utf-8 -*-
"""
Evaluation metrics test script
Usage: python tests/test_evaluation.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from evaluation.metrics import PointF1, PointAdjustF1, AUCMetrics, RangeBasedMetrics, compute_all_metrics


def test_evaluation_implementation():
    """Evaluation metrics implementation test"""

    print("=" * 60)
    print("🧪 Evaluation Metrics Test")
    print("=" * 60)

    all_passed = True

    # ==================== Test 1: Point F1 ====================
    print("\n[Test 1] Point-wise F1")
    print("-" * 40)

    # Simple test case
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    # TP=3 (index 2,3,7), FP=1 (index 6), FN=2 (index 4,8), TN=4

    try:
        point_f1 = PointF1()
        result = point_f1.compute(y_true, y_pred)

        print(f"✓ TP: {result['TP']}, FP: {result['FP']}, FN: {result['FN']}, TN: {result['TN']}")
        print(f"✓ Precision: {result['precision']:.4f}")
        print(f"✓ Recall: {result['recall']:.4f}")
        print(f"✓ F1: {result['f1']:.4f}")

        # Validate
        expected_tp, expected_fp, expected_fn = 3, 1, 2
        if result['TP'] == expected_tp and result['FP'] == expected_fp and result['FN'] == expected_fn:
            print("✅ Confusion matrix correct!")
        else:
            print(f"❌ Expected TP={expected_tp}, FP={expected_fp}, FN={expected_fn}")
            all_passed = False

        # Check precision = 3/(3+1) = 0.75
        if abs(result['precision'] - 0.75) < 0.01:
            print("✅ Precision correct!")
        else:
            print(f"❌ Expected precision=0.75, got {result['precision']}")
            all_passed = False

    except NotImplementedError as e:
        print(f"❌ Implementation needed: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 2: Point Adjustment F1 ====================
    print("\n[Test 2] Point Adjustment F1")
    print("-" * 40)

    # Test case: segment [2,3,4] has prediction at index 3
    #            segment [7,8] has prediction at index 7
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    # After adjustment: all points in detected segments become 1
    # Expected adjusted: [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]

    try:
        pa_f1 = PointAdjustF1()
        result = pa_f1.compute(y_true, y_pred)

        print(f"✓ Original predictions: {y_pred.tolist()}")
        print(f"✓ Adjusted predictions: {result['adjusted_predictions'].tolist()}")
        print(f"✓ Precision: {result['precision']:.4f}")
        print(f"✓ Recall: {result['recall']:.4f}")
        print(f"✓ F1: {result['f1']:.4f}")

        # After adjustment, all anomaly segments are detected
        # TP=5, FP=0, FN=0 -> Precision=1, Recall=1, F1=1
        if result['f1'] == 1.0:
            print("✅ Point Adjustment F1 = 1.0 (perfect after adjustment)!")
        else:
            print(f"⚠️ Expected F1=1.0 after adjustment, got {result['f1']}")

    except NotImplementedError as e:
        print(f"❌ Implementation needed: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 3: Edge Cases ====================
    print("\n[Test 3] Edge Cases")
    print("-" * 40)

    try:
        # All zeros
        y_true_zeros = np.array([0, 0, 0, 0, 0])
        y_pred_zeros = np.array([0, 0, 0, 0, 0])

        result_zeros = PointF1().compute(y_true_zeros, y_pred_zeros)
        print(f"✓ All zeros: Precision={result_zeros['precision']}, Recall={result_zeros['recall']}")

        # All ones
        y_true_ones = np.array([1, 1, 1, 1, 1])
        y_pred_ones = np.array([1, 1, 1, 1, 1])

        result_ones = PointF1().compute(y_true_ones, y_pred_ones)
        print(f"✓ All ones: Precision={result_ones['precision']}, Recall={result_ones['recall']}, F1={result_ones['f1']}")

        if result_ones['f1'] == 1.0:
            print("✅ Edge cases handled correctly!")
        else:
            all_passed = False

    except Exception as e:
        print(f"❌ Edge case error: {e}")
        all_passed = False

    # ==================== Test 4: AUC Metrics ====================
    print("\n[Test 4] AUC Metrics (ROC-AUC, PR-AUC)")
    print("-" * 40)

    try:
        # Create test data with anomaly scores
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
        # Higher scores for anomalies (good model)
        scores = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.8, 0.9, 0.85, 0.2, 0.1])

        auc_metrics = AUCMetrics()
        result = auc_metrics.compute(y_true, scores)

        print(f"✓ ROC-AUC: {result['roc_auc']:.4f}")
        print(f"✓ PR-AUC: {result['pr_auc']:.4f}")
        print(f"✓ Best threshold: {result['best_threshold']:.4f}")
        print(f"✓ Best F1 at threshold: {result['best_f1']:.4f}")

        # Good model should have high AUC
        if result['roc_auc'] > 0.8:
            print("✅ ROC-AUC > 0.8 (good separation)")
        else:
            print(f"⚠️ ROC-AUC={result['roc_auc']:.4f}, expected > 0.8")

    except NotImplementedError as e:
        print(f"❌ TODO(human) Implementation needed: {e}")
        print("   → src/evaluation/metrics.py의 AUCMetrics.compute() 메서드를 구현하세요")
        all_passed = False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 5: Range-based Metrics ====================
    print("\n[Test 5] Range-based Metrics")
    print("-" * 40)

    try:
        # Ground truth: segment [2,3,4], segment [7,8]
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        # Prediction: partial overlap with both segments
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0])

        range_metrics = RangeBasedMetrics()
        result = range_metrics.compute(y_true, y_pred)

        print(f"✓ Range Precision: {result['range_precision']:.4f}")
        print(f"✓ Range Recall: {result['range_recall']:.4f}")
        print(f"✓ Range F1: {result['range_f1']:.4f}")

        # Check that metrics are in valid range
        if 0 <= result['range_precision'] <= 1 and 0 <= result['range_recall'] <= 1:
            print("✅ Range-based metrics in valid range [0, 1]")
        else:
            print("❌ Metrics out of range")
            all_passed = False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 6: compute_all_metrics with scores ====================
    print("\n[Test 6] compute_all_metrics (with AUC)")
    print("-" * 40)

    try:
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
        scores = np.array([0.1, 0.2, 0.6, 0.8, 0.5, 0.2, 0.3, 0.7, 0.6, 0.1])

        all_results = compute_all_metrics(y_true, y_pred, scores=scores)

        print(f"✓ Point F1: {all_results['point_f1']['f1']:.4f}")
        print(f"✓ PA F1: {all_results['pa_f1']['f1']:.4f}")
        print(f"✓ Range F1: {all_results['range_based']['range_f1']:.4f}")

        # AUC metrics included when scores provided
        if 'auc' in all_results:
            print(f"✓ ROC-AUC: {all_results['auc']['roc_auc']:.4f}")
            print(f"✓ PR-AUC: {all_results['auc']['pr_auc']:.4f}")
            print("✅ All metrics computed successfully!")
        else:
            print("⚠️ AUC metrics not included (TODO(human) not implemented)")

        # PA F1 should be >= Point F1
        if all_results['pa_f1']['f1'] >= all_results['point_f1']['f1']:
            print("✅ PA F1 >= Point F1 (as expected)")
        else:
            print("⚠️ PA F1 should be >= Point F1")

    except NotImplementedError as e:
        print(f"⚠️ AUC metrics skipped (TODO(human) not implemented): {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        all_passed = False

    # ==================== Result ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Evaluation metrics implementation complete.")
    else:
        print("⚠️ Some tests failed. Check errors above.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_evaluation_implementation()
