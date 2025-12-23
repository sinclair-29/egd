#!/usr/bin/env python
"""
Quick verification script to test the repository setup
This runs a minimal attack with dummy data to verify all components work
"""

import sys
import traceback


def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False

    try:
        from tqdm import tqdm
        print(f"✓ tqdm")
    except ImportError as e:
        print(f"✗ tqdm import failed: {e}")
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_local_modules():
    """Test that local modules can be imported"""
    print("=" * 60)
    print("TEST 2: Checking local modules...")
    print("=" * 60)

    try:
        import config
        print(f"✓ config.py imported")
        print(f"  Device: {config.DEVICE}")
        print(f"  Default steps: {config.DEFAULT_NUM_STEPS}")
    except ImportError as e:
        print(f"✗ config.py import failed: {e}")
        return False

    try:
        import helper
        print(f"✓ helper.py imported")
    except ImportError as e:
        print(f"✗ helper.py import failed: {e}")
        return False

    try:
        import behavior
        print(f"✓ behavior.py imported")
    except ImportError as e:
        print(f"✗ behavior.py import failed: {e}")
        return False

    try:
        import attack
        print(f"✓ attack.py imported")
    except ImportError as e:
        print(f"✗ attack.py import failed: {e}")
        return False

    print("\n✓ All local modules imported successfully!\n")
    return True


def test_helper_functions():
    """Test helper functions with dummy data"""
    print("=" * 60)
    print("TEST 3: Testing helper functions...")
    print("=" * 60)

    try:
        import torch
        import helper

        # Test token creation
        device = "cpu"  # Use CPU for testing
        dummy_tokens = torch.tensor([1, 2, 3, 4, 5], device=device)
        print(f"✓ Created dummy tokens: {dummy_tokens}")

        # Test one-hot creation with dummy embeddings
        vocab_size = 100
        embed_dim = 64
        dummy_embed_weights = torch.randn(vocab_size, embed_dim, device=device)

        one_hot, embeddings = helper.create_one_hot_and_embeddings(
            dummy_tokens, dummy_embed_weights, device
        )
        print(f"✓ One-hot shape: {one_hot.shape}")
        print(f"✓ Embeddings shape: {embeddings.shape}")

    except Exception as e:
        print(f"✗ Helper function test failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ Helper functions working correctly!\n")
    return True


def test_optimizer():
    """Test the custom EGD optimizer"""
    print("=" * 60)
    print("TEST 4: Testing EGD optimizer...")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F
        from attack import EGDwithAdamOptimizer

        # Create dummy parameter (one-hot distribution)
        device = "cpu"
        param = F.softmax(torch.rand(5, 10, device=device), dim=1)
        param.requires_grad = True

        # Initialize optimizer
        optimizer = EGDwithAdamOptimizer([param], lr=0.01)
        print(f"✓ Optimizer initialized")

        # Simulate one optimization step
        loss = param.sum()  # Dummy loss
        loss.backward()
        optimizer.step()

        # Check that parameter is still on simplex
        row_sums = param.sum(dim=1)
        print(f"✓ Parameter row sums (should be ~1.0): {row_sums}")

        if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
            print(f"✓ Simplex constraint maintained")
        else:
            print(f"✗ Simplex constraint violated!")
            return False

    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ EGD optimizer working correctly!\n")
    return True


def test_data_loading():
    """Test that sample data can be loaded"""
    print("=" * 60)
    print("TEST 5: Testing data loading...")
    print("=" * 60)

    try:
        import helper

        # Try to load sample data
        behaviors = helper.parse_csv("./data/AdvBench/harmful_behaviors.csv")
        print(f"✓ Loaded {len(behaviors)} sample behaviors")

        if len(behaviors) > 0:
            print(f"✓ Sample behavior: {behaviors[0][0][:50]}...")

    except FileNotFoundError:
        print("⚠ Sample data file not found (this is okay if you haven't set up data yet)")
        return True  # Not a critical error
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ Data loading working correctly!\n")
    return True


def test_loss_calculation():
    """Test loss calculation with dummy model components"""
    print("=" * 60)
    print("TEST 6: Testing loss calculation...")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn
        from attack import calc_loss

        # Create a minimal dummy model
        class DummyModel:
            def __call__(self, inputs_embeds):
                class Output:
                    def __init__(self, logits):
                        self.logits = logits

                batch, seq_len, embed_dim = inputs_embeds.shape
                vocab_size = 100
                # Random logits
                logits = torch.randn(batch, seq_len, vocab_size)
                return Output(logits)

        model = DummyModel()
        device = "cpu"

        # Create dummy embeddings
        embed_dim = 64
        embeddings_user = torch.randn(1, 5, embed_dim)
        embeddings_adv = torch.randn(1, 3, embed_dim)
        embeddings_target = torch.randn(1, 4, embed_dim)
        targets = torch.randint(0, 100, (4,))

        # Calculate loss
        loss, logits = calc_loss(
            model, embeddings_user, embeddings_adv,
            embeddings_target, targets
        )

        print(f"✓ Loss calculated: {loss.item():.4f}")
        print(f"✓ Logits shape: {logits.shape}")

    except Exception as e:
        print(f"✗ Loss calculation test failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ Loss calculation working correctly!\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("REPOSITORY VERIFICATION TEST")
    print("=" * 60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Local Modules", test_local_modules),
        ("Helper Functions", test_helper_functions),
        ("EGD Optimizer", test_optimizer),
        ("Data Loading", test_data_loading),
        ("Loss Calculation", test_loss_calculation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✓✓✓ ALL TESTS PASSED! Repository is correctly set up. ✓✓✓")
        print("\nYou can now run the attack with:")
        print("  python run_attack.py --model_name Llama2 --dataset_name AdvBench --num_behaviors 1")
        return 0
    else:
        print(f"\n✗✗✗ {total - passed} test(s) failed. Please check the errors above. ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
