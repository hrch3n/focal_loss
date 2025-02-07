# Focal Loss for Binary Semantic Segmentation

## Overview
This project implements **Focal Loss** for binary semantic segmentation. The loss function is designed to handle class imbalance by focusing more on difficult examples, reducing the impact of easy-to-classify pixels.

## Features
- Supports **masking** to ignore certain pixels during loss computation.
- Implements **α-balancing** to adjust class contributions.
- Allows **γ parameter tuning** to focus on hard examples.
- Efficient **vectorized NumPy implementation** for scalability.
- CI/CD-ready with **pytest** for automated testing.

## Environment & Dependencies
This implementation was tested with the following setup:

- **Python Version:** 3.9+
- **Required Packages:**
  - `numpy==1.26.4`
  - `pytest==8.3.4`

## Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/hrch3n/focal_loss.git
cd focal_loss
pip install -r requirements.txt
```

Dependencies:
- `numpy`
- `pytest` (for testing)

## Usage
### Import the function
```python
import numpy as np
from focal_loss import loss_function

# Example input
y_pred = np.array([[0.0, 2.0], [-2.0, 1.0]])  # logits
y_true = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # binary ground truth

loss = loss_function(y_pred, y_true)
print(f"Focal Loss: {loss}")
```

### Function Parameters
| Parameter       | Description                                      | Default |
|---------------|------------------------------------------------|---------|
| `y_pred`      | Predicted logits or probability map (Height, Width) | Required |
| `y_true`      | Ground truth binary mask (Height, Width) | Required |
| `mask`        | Binary mask (1 = valid, 0 = ignored) | `None` |
| `alpha`       | Class weighting for class 1 | `0.25` |
| `gamma`       | Focusing parameter for hard examples | `2.0` |
| `apply_sigmoid` | Whether to apply `sigmoid()` to `y_pred` | `True` |

## Testing
Run the test suite using `pytest`:
```bash
pytest -s -v test_focal_loss.py
```

### Test Cases
| Test | Description |
|------|------------|
| **Mismatched Shapes** | Ensures function raises an error when `y_pred` and `y_true` have different shapes |
| **Basic Functionality** | Checks focal loss computation for standard cases |
| **Gamma = 0 (BCE Mode)** | Checks if focal loss behaves like binary cross-entropy |
| **Extreme Cases (0s & 1s)** | Verifies loss when all pixels belong to a single class |
| **Masking Support** | Ensures loss correctly ignores masked pixels |
| **Focal Loss vs BCE on Highly Imbalanced Data (1:1000)** | Ensures focal loss down-weights easy negatives effectively |
| **Zero Logits** | Confirms stability when logits are `0` |
| **Large Logits** | Tests numerical stability with logits `±100` |
| **Large Image Test (4096x4096)** | Ensures efficient computation for large images |
| **Invalid Mask Shape** | Verifies error handling for mismatched mask size |


## Scalability & Production Readiness
- **Fully vectorized** for fast execution on large images.
- **CI/CD-ready** with `pytest` for automated testing.
- **Error handling** for invalid inputs (`TypeError`, `ValueError`).

## Author
Developed by **Haoran Chen**
For questions or contributions, open an issue on GitHub.