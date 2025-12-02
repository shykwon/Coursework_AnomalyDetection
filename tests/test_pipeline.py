# -*- coding: utf-8 -*-
"""
TSAD Pipeline 구현 테스트 스크립트
사용법: python tests/test_pipeline.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from preprocessing.scaler import Scaler
from models.dlinear_model import DLinearModel
from pipeline.tsad_pipeline import TSADPipeline


def test_pipeline_implementation():
    """TSAD Pipeline 구현 테스트"""

    print("=" * 60)
    print("🧪 TSAD Pipeline 구현 테스트")
    print("=" * 60)

    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    t = np.linspace(0, 10 * np.pi, n_samples)

    # 정상 패턴: 정현파
    data = np.zeros((n_samples, n_features))
    for i in range(n_features):
        data[:, i] = np.sin(t + i * 0.5) * 10 + 50 + np.random.randn(n_samples) * 0.5

    # 이상치 주입
    anomaly_indices = [350, 351, 352, 400, 401]
    for idx in anomaly_indices:
        data[idx, :] += np.random.randn(n_features) * 20

    train_data = data[:300]
    test_data = data[300:]

    all_passed = True

    # ==================== 파이프라인 구성 테스트 ====================
    print("\n[테스트 1] 파이프라인 구성")
    print("-" * 40)

    try:
        scaler = Scaler(method='standard')
        model = DLinearModel(seq_len=50, pred_len=1, epochs=2, batch_size=32)
        pipeline = TSADPipeline(preprocessor=scaler, model=model)

        print("✓ Scaler 생성 완료")
        print("✓ DLinearModel 생성 완료")
        print("✓ TSADPipeline 생성 완료")
        print("✅ 파이프라인 구성 성공")

    except Exception as e:
        print(f"❌ 파이프라인 구성 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ==================== 파이프라인 fit 테스트 ====================
    print("\n[테스트 2] 파이프라인 fit")
    print("-" * 40)

    try:
        print("파이프라인 학습 시작...")
        pipeline.fit(train_data, verbose=True)

        print(f"✓ 자동 설정된 threshold: {pipeline.get_threshold():.6f}")
        print("✅ 파이프라인 fit 성공")

    except NotImplementedError as e:
        print(f"❌ 구현 필요: {e}")
        all_passed = False
        return False
    except Exception as e:
        print(f"❌ fit 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ==================== 파이프라인 predict 테스트 ====================
    print("\n[테스트 3] 파이프라인 predict")
    print("-" * 40)

    try:
        scores, labels = pipeline.predict(test_data)

        print(f"✓ 점수 shape: {scores.shape}")
        print(f"✓ 라벨 shape: {labels.shape}")
        print(f"✓ 점수 범위: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"✓ 탐지된 이상치 수: {labels.sum()}")

        if scores.shape == (len(test_data),) and labels.shape == (len(test_data),):
            print("✅ 파이프라인 predict 성공")
        else:
            print("❌ 출력 shape 불일치")
            all_passed = False

    except NotImplementedError as e:
        print(f"❌ 구현 필요: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ predict 오류: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== 전체 흐름 테스트 ====================
    print("\n[테스트 4] fit_predict 전체 흐름")
    print("-" * 40)

    try:
        # 새 파이프라인으로 fit_predict 테스트
        scaler2 = Scaler(method='minmax')
        model2 = DLinearModel(seq_len=30, pred_len=1, epochs=2)
        pipeline2 = TSADPipeline(preprocessor=scaler2, model=model2)

        scores2, labels2 = pipeline2.fit_predict(train_data, test_data, verbose=False)

        print(f"✓ fit_predict 완료")
        print(f"✓ 탐지된 이상치 비율: {labels2.mean() * 100:.2f}%")
        print("✅ 전체 흐름 테스트 성공")

    except Exception as e:
        print(f"❌ fit_predict 오류: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== 결과 ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 모든 테스트 통과! Pipeline 구현이 완료되었습니다.")
        print("\n다음 단계:")
        print("1. Scaler의 TODO(human) 구현")
        print("2. DLinearModel의 TODO(human) 구현")
        print("3. TSADPipeline의 TODO(human) 구현")
        print("4. 실제 PSM/SWaT 데이터로 테스트")
    else:
        print("⚠️ 일부 테스트 실패. 위의 오류를 확인하세요.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_pipeline_implementation()
