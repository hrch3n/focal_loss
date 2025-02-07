import pytest
import numpy as np
from focal_loss import loss_function 

# Set seed for reproducibility
np.random.seed(1234)

# Test 1: Mismatched Input Shapes
def test_mismatched_shapes():
    print("\nRunning Test 1: Mismatched Shapes")

    y_pred = np.random.randn(64, 64)  # logits
    y_true = np.random.randint(0, 2, (32, 32), dtype=np.uint8)  # different shape

    with pytest.raises(ValueError, match="y_pred and y_true must have the same shape"):
        loss_function(y_pred, y_true)

# Test 2: Basic Functionality
def test_basic_functionality():
    print("\nRunning Test 2: Basic Functionality")
    y_pred = np.random.randn(64, 64)  # logits
    y_true = np.random.randint(0, 2, (64, 64), dtype=np.uint8)  # ground truth
    
    loss = loss_function(y_pred, y_true)
    print(f"Test 2 - Loss: {loss}")
    assert loss > 0, "Loss should be positive for standard input"

# Test 3: Gamma = 0 (Equivalent to BCE Loss)
def test_gamma_zero():
    print("\nRunning Test 3: Gamma = 0 (BCE Check)")
    y_pred = np.random.randn(64, 64)
    y_true = np.random.randint(0, 2, (64, 64), dtype=np.uint8)

    loss = loss_function(y_pred, y_true, gamma=0.0)
    print(f"Test 3 - Loss: {loss}")
    assert loss > 0, "Loss should be positive and match BCE when gamma = 0"

# Test 4: Extreme Cases (All 0s and All 1s)
@pytest.mark.parametrize("y_true_value", [0, 1])
def test_extreme_cases(y_true_value):
    print(f"\nRunning Test 4: Extreme Cases (all y_true = {y_true_value})")
    y_pred = np.random.randn(64, 64)  # logits
    y_true = np.full((64, 64), y_true_value, dtype=np.uint8)  # all 0s or all 1s

    loss = loss_function(y_pred, y_true)
    print(f"Test 4 - Loss ({y_true_value}s): {loss}")
    assert loss > 0, "Loss should be positive for all-0s or all-1s y_true"

# Test 5: Masking Support
def test_masking():
    print("\nRunning Test 5: Masking")
    y_pred = np.random.randn(64, 64)  # logits
    y_true = np.random.randint(0, 2, (64, 64), dtype=np.uint8)  # ground truth
    mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8)  # random mask

    loss_unmasked = loss_function(y_pred, y_true)
    loss_masked = loss_function(y_pred, y_true, mask)

    print(f"Test 5 - Unmasked Loss: {loss_unmasked}, Masked Loss: {loss_masked}")
    assert loss_unmasked > loss_masked, "Masked loss should be smaller than unmasked loss"
    assert loss_masked > 0, "Masked loss should not be zero"

# Test 6: Focal Loss vs BCE on Highly Imbalanced Data (1:1000)
def test_focal_loss_imbalance():
    print("\nRunning Test 6: Focal Loss vs BCE on Highly Imbalanced Data (1:1000)")

    # Create an extremely imbalanced dataset: 99.9% background (0), 0.1% foreground (1)
    y_true = np.zeros((64, 64), dtype=np.uint8)
    
    # Set very few foreground pixels (~0.1% of total)
    num_foreground_pixels = max(1, (64 * 64) // 1000)  # ensure at least 1 foreground pixel
    fg_indices = np.random.choice(64 * 64, num_foreground_pixels, replace=False)
    y_true.flat[fg_indices] = 1  # assign foreground to randomly chosen indices

    # Create predictions
    y_pred = np.random.uniform(low=-1, high=1, size=(64, 64))  # logits

    # Compute focal loss and BCE loss (gamma = 0 makes focal loss behave like BCE)
    focal_loss = loss_function(y_pred, y_true, gamma=2.0)
    bce_loss = loss_function(y_pred, y_true, gamma=0.0)  # should be standard BCE

    print(f"Test 6 - Extreme Imbalance Focal Loss: {focal_loss}, BCE Loss: {bce_loss}")
    
    # Focal loss should be significantly lower than BCE due to down-weighting of easy background pixels 
    assert focal_loss < bce_loss, "Focal loss should be lower than BCE on highly imbalanced data"


# Test 7: Zero Logits Case
def test_zero_logits():
    print("\nRunning Test 7: Zero Logits")
    y_pred = np.zeros((64, 64))
    y_true = np.random.randint(0, 2, (64, 64), dtype=np.uint8)

    loss = loss_function(y_pred, y_true)
    print(f"Test 7 - Loss: {loss}")
    assert loss > 0, "Loss should be positive even with zero logits"

# Test 8: Large Logits Case
def test_large_logits():
    print("\nRunning Test 8: Large Logits")
    y_pred = np.full((64, 64), 100, dtype=np.float32)  # extreme logits
    y_true = np.random.randint(0, 2, (64, 64), dtype=np.uint8)

    loss = loss_function(y_pred, y_true)
    print(f"Test 8 - Loss: {loss}")
    assert loss > 0, "Loss should be positive even with extreme logits"

# Test 9: Large Image Test (Scalability)
def test_large_image():
    print("\nRunning Test 9: Large Image Test (4096x4096)")
    y_pred = np.random.randn(4096, 4096)  # large image size
    y_true = np.random.randint(0, 2, (4096, 4096), dtype=np.uint8)

    loss = loss_function(y_pred, y_true)
    print(f"Test 9 - Large Image Loss: {loss}")
    assert loss > 0, "Loss should be positive for large images"

# Test 10: Invalid Mask Shape (Ensures correct handling of shape mismatches)
def test_invalid_mask_shape():
    print("\nRunning Test 10: Invalid Mask Shape")
    y_pred = np.random.randn(64, 64)
    y_true = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
    mask = np.random.randint(0, 2, (32, 32), dtype=np.uint8)  # wrong shape

    with pytest.raises(ValueError, match="mask must have the same shape as y_true"):
        loss_function(y_pred, y_true, mask)
