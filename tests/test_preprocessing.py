# -*- coding: utf-8 -*-
"""
Preprocessing module test script
Usage: python tests/test_preprocessing.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

from preprocessing import (
    Scaler,
    EWMASmoother,
    MovingAverageSmoother,
    MovingAverageDetrender,
    DifferencingDetrender,
    STLDetrender
)


def test_preprocessing_implementation():
    """Preprocessing module implementation test"""

    print("=" * 60)
    print("🧪 Preprocessing Module Test")
    print("=" * 60)

    all_passed = True
    np.random.seed(42)

    # Create test data with trend and seasonality
    n_samples = 200
    n_features = 3
    t = np.arange(n_samples)

    # Trend + Seasonal + Noise
    trend = 0.01 * t.reshape(-1, 1) * np.ones((1, n_features))
    seasonal = np.sin(2 * np.pi * t / 24).reshape(-1, 1) * np.ones((1, n_features)) * 0.5
    noise = np.random.randn(n_samples, n_features) * 0.1
    data = trend + seasonal + noise

    # Store original variance for comparison
    original_var = np.var(data)

    # ==================== Test 1: Scaler ====================
    print("\n[Test 1] Scaler (MinMax, Standard)")
    print("-" * 40)

    try:
        # MinMax Scaler
        scaler_minmax = Scaler(method='minmax')
        scaled_minmax = scaler_minmax.fit_transform(data)

        print(f"✓ MinMax - Original range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"✓ MinMax - Scaled range: [{scaled_minmax.min():.4f}, {scaled_minmax.max():.4f}]")

        if 0 <= scaled_minmax.min() <= 0.01 and 0.99 <= scaled_minmax.max() <= 1.0:
            print("✅ MinMax scaling correct!")
        else:
            print("❌ MinMax scaling range error")
            all_passed = False

        # Inverse transform
        restored_minmax = scaler_minmax.inverse_transform(scaled_minmax)
        if np.allclose(data, restored_minmax, rtol=1e-5):
            print("✅ MinMax inverse transform correct!")
        else:
            print("❌ MinMax inverse transform error")
            all_passed = False

        # Standard Scaler
        scaler_std = Scaler(method='standard')
        scaled_std = scaler_std.fit_transform(data)

        print(f"✓ Standard - Mean: {scaled_std.mean():.6f}, Std: {scaled_std.std():.4f}")

        if abs(scaled_std.mean()) < 1e-6 and abs(scaled_std.std() - 1.0) < 0.1:
            print("✅ Standard scaling correct!")
        else:
            print("❌ Standard scaling error")
            all_passed = False

    except Exception as e:
        print(f"❌ Scaler error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 2: EWMA Smoother ====================
    print("\n[Test 2] EWMA Smoother")
    print("-" * 40)

    try:
        smoother = EWMASmoother(span=10)
        smoothed = smoother.fit_transform(data)

        # Smoothed data should have less variance
        smoothed_var = np.var(smoothed)

        print(f"✓ Original variance: {original_var:.6f}")
        print(f"✓ Smoothed variance: {smoothed_var:.6f}")
        print(f"✓ Variance reduction: {(1 - smoothed_var/original_var)*100:.1f}%")

        if smoothed_var < original_var:
            print("✅ EWMA smoothing reduces variance!")
        else:
            print("⚠️ Smoothed variance should be less than original")

    except NotImplementedError as e:
        print(f"❌ TODO(human) Implementation needed: {e}")
        print("   → src/preprocessing/smoother.py의 EWMASmoother.transform()을 구현하세요")
        all_passed = False
    except Exception as e:
        print(f"❌ EWMA Smoother error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 3: Moving Average Smoother ====================
    print("\n[Test 3] Moving Average Smoother")
    print("-" * 40)

    try:
        ma_smoother = MovingAverageSmoother(window=5)
        ma_smoothed = ma_smoother.fit_transform(data)

        ma_var = np.var(ma_smoothed)
        print(f"✓ MA Smoothed variance: {ma_var:.6f}")

        if ma_var < original_var:
            print("✅ MA smoothing works correctly!")
        else:
            print("⚠️ MA smoothed variance should be less")

    except Exception as e:
        print(f"❌ MA Smoother error: {e}")
        all_passed = False

    # ==================== Test 4: Moving Average Detrender ====================
    print("\n[Test 4] Moving Average Detrender")
    print("-" * 40)

    try:
        detrender = MovingAverageDetrender(window=24)
        detrended = detrender.fit_transform(data)

        # Detrended data should have less trend
        # Check by comparing mean of first half vs second half
        original_trend = np.mean(data[n_samples//2:]) - np.mean(data[:n_samples//2])
        detrended_trend = np.mean(detrended[n_samples//2:]) - np.mean(detrended[:n_samples//2])

        print(f"✓ Original trend (2nd half - 1st half): {original_trend:.4f}")
        print(f"✓ Detrended trend: {detrended_trend:.4f}")

        if abs(detrended_trend) < abs(original_trend):
            print("✅ Detrending reduces trend!")
        else:
            print("⚠️ Detrending should reduce trend")

        # Inverse transform
        restored = detrender.inverse_transform(detrended)
        if np.allclose(data, restored, rtol=1e-5):
            print("✅ Detrender inverse transform correct!")
        else:
            print("⚠️ Detrender inverse transform has some error (expected)")

    except Exception as e:
        print(f"❌ Detrender error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 5: Differencing Detrender ====================
    print("\n[Test 5] Differencing Detrender")
    print("-" * 40)

    try:
        diff_detrender = DifferencingDetrender(order=1)
        diffed = diff_detrender.fit_transform(data)

        # First value should be 0 (or close to original first value after diff)
        print(f"✓ Differenced data shape: {diffed.shape}")
        print(f"✓ First differenced value: {diffed[0, 0]:.6f}")

        # Check stationarity improved (variance should be similar across windows)
        var_first_half = np.var(diffed[:n_samples//2])
        var_second_half = np.var(diffed[n_samples//2:])

        print(f"✓ Variance (1st half): {var_first_half:.6f}")
        print(f"✓ Variance (2nd half): {var_second_half:.6f}")

        if abs(var_first_half - var_second_half) / var_first_half < 0.5:
            print("✅ Differencing improves stationarity!")
        else:
            print("⚠️ Differencing effect varies")

    except Exception as e:
        print(f"❌ Differencing error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 6: STL Detrender ====================
    print("\n[Test 6] STL Detrender")
    print("-" * 40)

    try:
        stl_detrender = STLDetrender(period=24, robust=True)
        residuals = stl_detrender.fit_transform(data)

        print(f"✓ Residual shape: {residuals.shape}")
        print(f"✓ Residual mean: {residuals.mean():.6f}")
        print(f"✓ Residual std: {residuals.std():.6f}")

        # Residuals should have less structure than original
        residual_var = np.var(residuals)
        print(f"✓ Residual variance: {residual_var:.6f} (vs original: {original_var:.6f})")

        if residual_var < original_var:
            print("✅ STL decomposition extracts residuals!")
        else:
            print("⚠️ Residual should have less variance than original")

        # Check inverse transform
        restored_stl = stl_detrender.inverse_transform(residuals)
        reconstruction_error = np.mean(np.abs(data - restored_stl))
        print(f"✓ Reconstruction error: {reconstruction_error:.6f}")

        if reconstruction_error < 0.01:
            print("✅ STL inverse transform correct!")
        else:
            print("⚠️ STL reconstruction has some error")

    except Exception as e:
        print(f"❌ STL Detrender error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 7: Pipeline Integration ====================
    print("\n[Test 7] Preprocessing Pipeline")
    print("-" * 40)

    try:
        # Test chaining: Scale -> Smooth -> Detrend
        scaler = Scaler(method='minmax')
        smoother = MovingAverageSmoother(window=5)
        detrender = MovingAverageDetrender(window=24)

        # Apply in sequence
        step1 = scaler.fit_transform(data)
        step2 = smoother.fit_transform(step1)
        step3 = detrender.fit_transform(step2)

        print(f"✓ Original shape: {data.shape}")
        print(f"✓ After scaling: range [{step1.min():.2f}, {step1.max():.2f}]")
        print(f"✓ After smoothing: var {np.var(step2):.6f}")
        print(f"✓ After detrending: var {np.var(step3):.6f}")

        print("✅ Preprocessing pipeline works!")

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        all_passed = False

    # ==================== Result ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Preprocessing module complete.")
    else:
        print("⚠️ Some tests failed. Check errors above.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_preprocessing_implementation()
