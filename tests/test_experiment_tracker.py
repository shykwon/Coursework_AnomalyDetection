# -*- coding: utf-8 -*-
"""
ExperimentTracker test script
Usage: python tests/test_experiment_tracker.py
"""

import sys
import numpy as np
import shutil
from pathlib import Path

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from utils.experiment_tracker import ExperimentTracker


def test_experiment_tracker():
    """ExperimentTracker implementation test"""

    print("=" * 60)
    print("🧪 ExperimentTracker Test")
    print("=" * 60)

    # 테스트용 임시 디렉토리
    test_dir = Path('/tmp/test_experiment_tracker')
    if test_dir.exists():
        shutil.rmtree(test_dir)

    all_passed = True

    try:
        # ==================== Test 1: 초기화 ====================
        print("\n[Test 1] Initialization")
        print("-" * 40)

        tracker = ExperimentTracker(base_dir=str(test_dir))

        assert test_dir.exists()
        assert (test_dir / 'scores').exists()
        assert (test_dir / 'results').exists()

        print("✅ Directories created successfully!")

        # ==================== Test 2: Step 1 - 학습 기록 ====================
        print("\n[Test 2] Step 1 - Training Log")
        print("-" * 40)

        # 가상의 Score 생성
        fake_scores_1 = np.random.randn(1000) * 0.5 + 1.0
        fake_scores_2 = np.random.randn(1000) * 0.3 + 0.8

        # 학습 결과 기록
        tracker.log_training(
            model='DLinear',
            preprocess='P_MM',
            dataset='PSM',
            scores=fake_scores_1,
            training_time=120.5,
            config={'seq_len': 96, 'pred_len': 1}
        )

        tracker.log_training(
            model='OmniAnomaly',
            preprocess='P_MM',
            dataset='PSM',
            scores=fake_scores_2,
            training_time=250.3
        )

        # 파일 확인
        assert (test_dir / 'scores' / 'DLinear_P_MM_PSM.npy').exists()
        assert (test_dir / 'scores' / 'OmniAnomaly_P_MM_PSM.npy').exists()
        assert (test_dir / 'results' / 'training_log.csv').exists()

        print("✅ Training logs saved successfully!")

        # ==================== Test 3: Score 로드 ====================
        print("\n[Test 3] Load Scores")
        print("-" * 40)

        loaded_scores = tracker.load_scores('DLinear', 'P_MM', 'PSM')
        assert len(loaded_scores) == 1000
        assert np.allclose(loaded_scores, fake_scores_1)

        available = tracker.list_available_scores()
        print(f"✓ Available scores: {len(available)}")
        for s in available:
            print(f"  - {s['model']}_{s['preprocess']}_{s['dataset']}")

        print("✅ Score loading works!")

        # ==================== Test 4: Step 2 - 평가 기록 ====================
        print("\n[Test 4] Step 2 - Evaluation Log")
        print("-" * 40)

        # 여러 후처리 결과 기록
        postprocesses = ['T1', 'T2', 'T3', 'T4', 'T5']
        for i, postprocess in enumerate(postprocesses):
            tracker.log_evaluation(
                model='DLinear',
                preprocess='P_MM',
                postprocess=postprocess,
                dataset='PSM',
                metrics={
                    'point_f1': 0.70 + i * 0.02,
                    'pa_f1': 0.80 + i * 0.03,
                    'roc_auc': 0.85 + i * 0.02,
                    'pr_auc': 0.75 + i * 0.02
                },
                threshold_value=0.5 + i * 0.1
            )

        # OmniAnomaly 결과도 추가
        for i, postprocess in enumerate(postprocesses):
            tracker.log_evaluation(
                model='OmniAnomaly',
                preprocess='P_MM',
                postprocess=postprocess,
                dataset='PSM',
                metrics={
                    'point_f1': 0.68 + i * 0.02,
                    'pa_f1': 0.78 + i * 0.03,
                    'roc_auc': 0.83 + i * 0.02,
                    'pr_auc': 0.73 + i * 0.02
                }
            )

        assert (test_dir / 'results' / 'evaluation_results.csv').exists()
        print("✅ Evaluation logs saved successfully!")

        # ==================== Test 5: 보고서 생성 ====================
        print("\n[Test 5] Generate Report")
        print("-" * 40)

        report = tracker.generate_report()

        print("\n📊 Full Results:")
        print(report['full'][['model', 'preprocess', 'postprocess', 'pa_f1']].head(10).to_string(index=False))

        print("\n📊 Postprocess Comparison (PA F1):")
        print(report['postprocess_comparison'].to_string())

        print("\n📊 Best Combinations:")
        print(report['best_combinations'].to_string(index=False))

        print("\n✅ Report generation works!")

        # ==================== Test 6: 최적 조합 ====================
        print("\n[Test 6] Get Best Combination")
        print("-" * 40)

        best = tracker.get_best(metric='pa_f1')
        print(f"✓ Best overall: {best['model']} + {best['preprocess']} + {best['postprocess']}")
        print(f"  PA F1 = {best['pa_f1']:.4f}")

        best_dlinear = tracker.get_best(metric='pa_f1', model='DLinear')
        print(f"✓ Best DLinear: {best_dlinear['postprocess']}, PA F1 = {best_dlinear['pa_f1']:.4f}")

        print("✅ Best combination lookup works!")

        # ==================== Test 7: Summary ====================
        print("\n[Test 7] Print Summary")
        print("-" * 40)

        tracker.print_summary()

        # ==================== Test 8: Excel Export ====================
        print("\n[Test 8] Excel Export")
        print("-" * 40)

        tracker.export_to_excel('test_results.xlsx')
        assert (test_dir / 'results' / 'test_results.xlsx').exists()
        print("✅ Excel export works!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    finally:
        # 테스트 디렉토리 정리
        if test_dir.exists():
            shutil.rmtree(test_dir)

    # ==================== Result ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! ExperimentTracker is ready.")
    else:
        print("⚠️ Some tests failed. Check errors above.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_experiment_tracker()
