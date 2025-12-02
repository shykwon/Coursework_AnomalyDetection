# -*- coding: utf-8 -*-
"""
OmniAnomaly model test script
Usage: python tests/test_omnianomaly.py
"""

import sys
import numpy as np

sys.path.insert(0, '/home/sheda7788/project/Anomaly-Detection/src')

import torch
from models.cores.omnianomaly_core import OmniAnomalyCore
from models.omnianomaly_model import OmniAnomaly


def test_omnianomaly_implementation():
    """OmniAnomaly model implementation test"""

    print("=" * 60)
    print("🧪 OmniAnomaly Model Test")
    print("=" * 60)

    all_passed = True
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Device: {device}")

    # Test parameters
    batch_size = 16
    seq_len = 50
    x_dim = 10
    z_dim = 4

    # ==================== Test 1: Core Model Forward ====================
    print("\n[Test 1] OmniAnomalyCore Forward Pass")
    print("-" * 40)

    try:
        core = OmniAnomalyCore(
            x_dim=x_dim,
            z_dim=z_dim,
            hidden_dim=32,
            n_flows=2,
            use_flow=True
        ).to(device)

        # Random input
        x = torch.randn(batch_size, seq_len, x_dim).to(device)

        # Forward
        outputs = core(x)

        print(f"✓ Input shape: {x.shape}")
        print(f"✓ x_mu shape: {outputs['x_mu'].shape}")
        print(f"✓ z shape: {outputs['z'].shape}")
        print(f"✓ z_mu shape: {outputs['z_mu'].shape}")

        # Check shapes
        assert outputs['x_mu'].shape == (batch_size, seq_len, x_dim)
        assert outputs['z'].shape == (batch_size, seq_len, z_dim)

        print("✅ Core forward pass correct!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 2: Loss Computation ====================
    print("\n[Test 2] Loss Computation")
    print("-" * 40)

    try:
        losses = core.compute_loss(x, outputs, beta=1.0)

        print(f"✓ Total loss: {losses['loss'].item():.4f}")
        print(f"✓ Recon loss: {losses['recon_loss'].item():.4f}")
        print(f"✓ KL loss: {losses['kl_loss'].item():.4f}")

        # Loss should be positive
        assert losses['loss'].item() > 0
        assert losses['recon_loss'].item() > 0

        print("✅ Loss computation correct!")

    except NotImplementedError as e:
        print(f"❌ TODO(human) Implementation needed: {e}")
        print("   → src/models/cores/omnianomaly_core.py의 compute_loss()를 구현하세요")
        all_passed = False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 3: Anomaly Score ====================
    print("\n[Test 3] Anomaly Score Calculation")
    print("-" * 40)

    try:
        scores = core.get_anomaly_score(x, n_samples=1, last_point_only=True)

        print(f"✓ Scores shape: {scores.shape}")
        print(f"✓ Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

        assert scores.shape == (batch_size,)

        print("✅ Anomaly score calculation correct!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 4: Wrapper Model ====================
    print("\n[Test 4] OmniAnomaly Wrapper")
    print("-" * 40)

    try:
        # Create synthetic data with anomalies
        n_samples = 500
        n_features = 5

        # Normal data
        normal_data = np.random.randn(n_samples, n_features) * 0.5

        # Add trend
        trend = np.linspace(0, 1, n_samples).reshape(-1, 1)
        train_data = normal_data + trend

        # Test data with anomalies
        test_data = np.random.randn(n_samples, n_features) * 0.5 + trend
        # Inject anomalies at indices 200-210
        test_data[200:210] = test_data[200:210] + 5.0

        # Model config
        config = {
            'x_dim': n_features,
            'z_dim': 4,
            'hidden_dim': 32,
            'window_length': 50,
            'n_flows': 2,
            'epochs': 3,  # Quick test
            'batch_size': 32,
            'learning_rate': 1e-3,
            'device': str(device)
        }

        model = OmniAnomaly(config)

        print("✓ Model created")
        print(f"✓ Training with {n_samples} samples...")

        # Fit (quick test)
        model.fit(train_data)
        print("✓ Model fitted")

        # Get anomaly scores
        scores = model.get_anomaly_score(test_data)
        print(f"✓ Anomaly scores shape: {scores.shape}")
        print(f"✓ Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Check if anomaly region has higher scores
        normal_scores = np.concatenate([scores[:190], scores[220:]])
        anomaly_scores = scores[200:210]

        print(f"✓ Normal region mean score: {normal_scores.mean():.4f}")
        print(f"✓ Anomaly region mean score: {anomaly_scores.mean():.4f}")

        if anomaly_scores.mean() > normal_scores.mean():
            print("✅ Anomaly region has higher scores (as expected)!")
        else:
            print("⚠️ Anomaly detection may need more training")

        print("✅ Wrapper model works!")

    except NotImplementedError as e:
        print(f"❌ TODO(human) Implementation needed: {e}")
        all_passed = False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Test 5: Planar Flow ====================
    print("\n[Test 5] Normalizing Flow")
    print("-" * 40)

    try:
        from models.cores.omnianomaly_core import PlanarFlow, NormalizingFlows

        flow = PlanarFlow(z_dim=z_dim).to(device)
        z = torch.randn(batch_size, z_dim).to(device)

        z_new, log_det = flow(z)

        print(f"✓ Input z shape: {z.shape}")
        print(f"✓ Output z shape: {z_new.shape}")
        print(f"✓ Log determinant shape: {log_det.shape}")

        # Check invertibility property (log_det should be finite)
        assert torch.isfinite(log_det).all()

        # Test stacked flows
        flows = NormalizingFlows(z_dim=z_dim, n_flows=5).to(device)
        z_k, sum_log_det = flows(z)

        print(f"✓ Stacked flows output shape: {z_k.shape}")
        print(f"✓ Sum log det: {sum_log_det.mean().item():.4f}")

        print("✅ Normalizing Flow works!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # ==================== Result ====================
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! OmniAnomaly implementation complete.")
    else:
        print("⚠️ Some tests failed. Check errors above.")
    print("=" * 60)

    return all_passed


if __name__ == '__main__':
    test_omnianomaly_implementation()
