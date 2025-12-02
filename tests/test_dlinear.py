# -*- coding: utf-8 -*-
"""
DLinear 모델 구현 테스트 스크립트
사용법: python tests/test_dlinear.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from models.dlinear_model import DLinearModel


def test_dlinear_implementation():
    """DLinear 모델 구현 테스트"""

    print("=" * 60)
    print("🧪 DLinear 모델 구현 테스트")
    print("=" * 60)

    # 테스트 데이터 생성 (정현파 + 노이즈)
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    t = np.linspace(0, 10 * np.pi, n_samples)

    # 정상 패턴: 정현파
    data = np.zeros((n_samples, n_features))
    for i in range(n_features):
        data[:, i] = np.sin(t + i * 0.5) + np.random.randn(n_samples) * 0.1

    # 이상치 주입 (일부 구간에 스파이크)
    anomaly_indices = [200, 201, 202, 350, 351]
    for idx in anomaly_indices:
        data[idx, :] += np.random.randn(n_features) * 3

    train_data = data[:300]
    test_data = data[300:]

    all_passed = True

    # ==================== 시퀀스 생성 테스트 ====================
    print("\n[테스트 1] 슬라이딩 윈도우 시퀀스 생성")
    print("-" * 40)

    try:
        model = DLinearModel(seq_len=50, pred_len=1, epochs=1)
        X, y = model._create_sequences(train_data)

        print(f"✓ 입력 데이터 shape: {train_data.shape}")
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")

        expected_n_seq = len(train_data) - 50 - 1 + 1  # 250
        if X.shape == (expected_n_seq, 50, n_features):
            print("✅ 시퀀스 shape 정상")
        else:
            print(f"❌ 예상 shape: ({expected_n_seq}, 50, {n_features})")
            all_passed = False

    except NotImplementedError as e:
        print(f"❌ 구현 필요: {e}")
        all_passed = False
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        return False

    # ==================== 모델 학습 테스트 ====================
    print("\n[테스트 2] 모델 학습")
    print("-" * 40)

    try:
        model = DLinearModel(
            seq_len=50,
            pred_len=1,
            epochs=3,
            batch_size=32,
            lr=0.001
        )

        print("학습 시작...")
        model.fit(train_data, verbose=True)
        print("✅ 모델 학습 완료")

    except Exception as e:
        print(f"❌ 학습 오류: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
        return False

    # ==================== 예측 테스트 ====================
    print("\n[테스트 3] 예측 수행")
    print("-" * 40)

    try:
        predictions = model.predict(test_data)
        print(f"✓ 예측 shape: {predictions.shape}")
        print(f"✓ 예측값 범위: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print("✅ 예측 정상")

    except Exception as e:
        print(f"❌ 예측 오류: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== 이상치 점수 테스트 ====================
    print("\n[테스트 4] 이상치 점수 계산")
    print("-" * 40)

    try:
        scores = model.get_anomaly_score(test_data)

        print(f"✓ 점수 shape: {scores.shape}")
        print(f"✓ 점수 범위: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"✓ 점수 평균: {scores.mean():.4f}")

        # 이상치 구간(인덱스 50, 51 = 원본 350, 351)의 점수가 높은지 확인
        anomaly_region_scores = scores[45:60]  # 이상치 주변
        normal_region_scores = scores[100:150]  # 정상 구간

        print(f"✓ 이상치 구간 점수 평균: {anomaly_region_scores.mean():.4f}")
        print(f"✓ 정상 구간 점수 평균: {normal_region_scores.mean():.4f}")

        if scores.shape == (len(test_data),):
            print("✅ 점수 shape 정상")
        else:
            print(f"❌ 예상 shape: ({len(test_data)},)")
            all_passed = False

    except NotImplementedError as e:
        print(f"❌ 구현 필요: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ 점수 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== 결과 ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 모든 테스트 통과! DLinear 모델 구현이 완료되었습니다.")
    else:
        print("⚠️ 일부 테스트 실패. 위의 오류를 확인하세요.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_dlinear_implementation()
