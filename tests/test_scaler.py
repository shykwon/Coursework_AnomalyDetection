# -*- coding: utf-8 -*-
"""
Scaler 구현 테스트 스크립트
사용법: python tests/test_scaler.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from preprocessing.scaler import Scaler


def test_scaler_implementation():
    """Scaler 구현 테스트"""

    print("=" * 60)
    print("🧪 Scaler 구현 테스트")
    print("=" * 60)

    # 테스트 데이터 생성
    np.random.seed(42)
    train_data = np.random.randn(100, 5) * 10 + 50  # 평균 50, 표준편차 10
    test_data = np.random.randn(20, 5) * 10 + 50

    all_passed = True

    # ==================== MinMax Scaler 테스트 ====================
    print("\n[테스트 1] MinMax Scaler")
    print("-" * 40)

    try:
        scaler_minmax = Scaler(method='minmax')
        scaled_train = scaler_minmax.fit_transform(train_data)
        scaled_test = scaler_minmax.transform(test_data)

        # 검증 1: fit 파라미터
        params = scaler_minmax.get_params()
        if params['min'] is None or params['max'] is None:
            print("❌ fit()에서 min/max가 계산되지 않음")
            all_passed = False
        else:
            print(f"✓ min shape: {params['min'].shape}")
            print(f"✓ max shape: {params['max'].shape}")

        # 검증 2: transform 결과 범위
        if scaled_train is None:
            print("❌ transform() 결과가 None")
            all_passed = False
        else:
            min_val = scaled_train.min()
            max_val = scaled_train.max()
            print(f"✓ 스케일링 후 범위: [{min_val:.4f}, {max_val:.4f}]")

            if not (min_val >= -0.01 and max_val <= 1.01):
                print("⚠️ MinMax 결과가 [0, 1] 범위를 벗어남")
            else:
                print("✅ MinMax 범위 정상")

        # 검증 3: inverse_transform
        restored = scaler_minmax.inverse_transform(scaled_train)
        if restored is None:
            print("❌ inverse_transform() 결과가 None")
            all_passed = False
        else:
            diff = np.abs(restored - train_data).max()
            print(f"✓ 복원 오차 (max): {diff:.6f}")
            if diff < 0.01:
                print("✅ inverse_transform 정상")
            else:
                print("⚠️ 복원 오차가 큼")

    except NotImplementedError as e:
        print(f"❌ 구현 필요: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Standard Scaler 테스트 ====================
    print("\n[테스트 2] Standard Scaler")
    print("-" * 40)

    try:
        scaler_std = Scaler(method='standard')
        scaled_train = scaler_std.fit_transform(train_data)
        scaled_test = scaler_std.transform(test_data)

        # 검증 1: fit 파라미터
        params = scaler_std.get_params()
        if params['mean'] is None or params['std'] is None:
            print("❌ fit()에서 mean/std가 계산되지 않음")
            all_passed = False
        else:
            print(f"✓ mean shape: {params['mean'].shape}")
            print(f"✓ std shape: {params['std'].shape}")

        # 검증 2: transform 결과 통계
        if scaled_train is None:
            print("❌ transform() 결과가 None")
            all_passed = False
        else:
            mean_after = scaled_train.mean(axis=0)
            std_after = scaled_train.std(axis=0)
            print(f"✓ 스케일링 후 평균: {mean_after.mean():.4f} (목표: ~0)")
            print(f"✓ 스케일링 후 표준편차: {std_after.mean():.4f} (목표: ~1)")

            if np.abs(mean_after.mean()) < 0.1 and np.abs(std_after.mean() - 1) < 0.1:
                print("✅ Standard 스케일링 정상")
            else:
                print("⚠️ 평균/표준편차가 예상과 다름")

        # 검증 3: inverse_transform
        restored = scaler_std.inverse_transform(scaled_train)
        if restored is None:
            print("❌ inverse_transform() 결과가 None")
            all_passed = False
        else:
            diff = np.abs(restored - train_data).max()
            print(f"✓ 복원 오차 (max): {diff:.6f}")
            if diff < 0.01:
                print("✅ inverse_transform 정상")
            else:
                print("⚠️ 복원 오차가 큼")

    except NotImplementedError as e:
        print(f"❌ 구현 필요: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== fit 전 transform 호출 테스트 ====================
    print("\n[테스트 3] fit 전 transform 호출 시 에러")
    print("-" * 40)

    try:
        scaler_new = Scaler(method='standard')
        scaler_new.transform(train_data)  # fit 없이 transform
        print("❌ RuntimeError가 발생해야 하는데 발생하지 않음")
        all_passed = False
    except RuntimeError as e:
        print(f"✅ 예상대로 에러 발생: {str(e)[:50]}...")
    except Exception as e:
        print(f"⚠️ 다른 에러 발생: {e}")

    # ==================== 결과 ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 모든 테스트 통과! Scaler 구현이 정상적으로 완료되었습니다.")
    else:
        print("⚠️ 일부 테스트 실패. 위의 오류를 확인하세요.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_scaler_implementation()
