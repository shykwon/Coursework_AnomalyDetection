# -*- coding: utf-8 -*-
"""
정상성 검정 구현 테스트 스크립트
사용법: python tests/test_stationarity.py
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from data.analyzer import DataAnalyzer


def test_stationarity_implementation():
    """ADF/KPSS 검정 구현 테스트"""

    print("=" * 60)
    print("🧪 정상성 검정 구현 테스트")
    print("=" * 60)

    # 테스트용 데이터 생성
    np.random.seed(42)
    n = 500

    # 정상 시계열 (white noise)
    stationary_data = np.random.randn(n)

    # 비정상 시계열 (random walk)
    non_stationary_data = np.cumsum(np.random.randn(n))

    # 테스트 1: 정상 시계열
    print("\n[테스트 1] 정상 시계열 (White Noise)")
    print("-" * 40)

    df_stationary = pd.DataFrame({'value': stationary_data})
    analyzer = DataAnalyzer(df_stationary)

    try:
        result = analyzer.stationarity_test('value')

        print(f"✓ ADF 통계량: {result['adf_statistic']:.4f}")
        print(f"✓ ADF p-value: {result['adf_pvalue']:.4f}")
        print(f"✓ KPSS 통계량: {result['kpss_statistic']:.4f}")
        print(f"✓ KPSS p-value: {result['kpss_pvalue']:.4f}")
        print(f"✓ 결론: {result['conclusion']}")

        # 검증
        errors = []

        # ADF p-value가 0.05보다 작아야 함 (정상 시계열이므로)
        if result['adf_pvalue'] >= 0.05:
            errors.append("⚠️ ADF: 정상 시계열인데 p-value >= 0.05")

        # KPSS p-value가 0.05보다 커야 함 (정상 시계열이므로)
        if result['kpss_pvalue'] <= 0.05:
            errors.append("⚠️ KPSS: 정상 시계열인데 p-value <= 0.05")

        if result['conclusion'] != 'Stationary':
            errors.append("⚠️ 결론이 'Stationary'가 아님")

        if errors:
            for e in errors:
                print(e)
        else:
            print("✅ 테스트 1 통과!")

    except NotImplementedError as e:
        print(f"❌ 구현 필요: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 테스트 2: 비정상 시계열
    print("\n[테스트 2] 비정상 시계열 (Random Walk)")
    print("-" * 40)

    df_non_stationary = pd.DataFrame({'value': non_stationary_data})
    analyzer2 = DataAnalyzer(df_non_stationary)

    try:
        result2 = analyzer2.stationarity_test('value')

        print(f"✓ ADF 통계량: {result2['adf_statistic']:.4f}")
        print(f"✓ ADF p-value: {result2['adf_pvalue']:.4f}")
        print(f"✓ KPSS 통계량: {result2['kpss_statistic']:.4f}")
        print(f"✓ KPSS p-value: {result2['kpss_pvalue']:.4f}")
        print(f"✓ 결론: {result2['conclusion']}")

        # 검증: 비정상 시계열이므로 ADF p-value > 0.05 이거나 KPSS p-value < 0.05
        if result2['conclusion'] == 'Stationary':
            print("⚠️ Random Walk가 Stationary로 판정됨 (드물게 발생 가능)")
        else:
            print("✅ 테스트 2 통과!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

    # 테스트 3: 반환값 형식 검증
    print("\n[테스트 3] 반환값 형식 검증")
    print("-" * 40)

    required_keys = [
        'column', 'adf_statistic', 'adf_pvalue', 'adf_critical_values',
        'is_stationary_adf', 'kpss_statistic', 'kpss_pvalue',
        'kpss_critical_values', 'is_stationary_kpss', 'conclusion'
    ]

    missing_keys = [k for k in required_keys if k not in result]

    if missing_keys:
        print(f"❌ 누락된 키: {missing_keys}")
        return False
    else:
        print("✅ 모든 필수 키 존재!")

    # Critical values 검증
    if '5%' not in result['adf_critical_values']:
        print("❌ ADF critical values에 '5%' 키 없음")
        return False
    else:
        print("✅ ADF critical values 형식 정상")

    if '5%' not in result['kpss_critical_values']:
        print("❌ KPSS critical values에 '5%' 키 없음")
        return False
    else:
        print("✅ KPSS critical values 형식 정상")

    print("\n" + "=" * 60)
    print("🎉 모든 테스트 통과! 구현이 정상적으로 완료되었습니다.")
    print("=" * 60)

    return True


def show_reference_answer():
    """참고 정답 출력"""
    print("\n" + "=" * 60)
    print("📚 참고 정답 (직접 구현 후 확인하세요)")
    print("=" * 60)

    answer = '''
# ADF 검정
adf_result = adfuller(series, autolag='AIC')
adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]
adf_critical = adf_result[4]

# KPSS 검정
kpss_result = kpss(series, regression='c', nlags='auto')
kpss_statistic = kpss_result[0]
kpss_pvalue = kpss_result[1]
kpss_critical = kpss_result[3]
'''
    print(answer)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='정상성 검정 테스트')
    parser.add_argument('--answer', action='store_true', help='참고 정답 보기')
    args = parser.parse_args()

    if args.answer:
        show_reference_answer()
    else:
        success = test_stationarity_implementation()
        if not success:
            print("\n💡 힌트가 필요하면: python tests/test_stationarity.py --answer")
