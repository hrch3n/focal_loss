import numpy as np

def loss_function(y_pred, y_true, mask=None, alpha=0.25, gamma=2.0, apply_sigmoid=True):
    
    """
    An implementation of the Focal Loss for binary semantic segmentation.

    Reference:
    https://arxiv.org/pdf/1708.02002

    Parameters:
    - y_pred (numpy array): Predicted logits or probability map with shape (Height, Width).
                            If `apply_sigmoid=True`, `y_pred` should be raw logits.
                            If `apply_sigmoid=False`, `y_pred` should be probabilities (0 to 1).
    - y_true (numpy array): Ground truth binary mask of shape (Height, Width), with values in {0, 1}.
    - mask (numpy array, optional): Binary mask of shape (Height, Width), where 1 indicates valid pixels
                                    and 0 indicates pixels to ignore in loss computation. Default is None.
    - alpha (float, optional): Class weighting factor to balance class 1 and 0 contributions.
                               If `alpha` is in (0,1), it applies different weights to 1 and 0 samples.
                               Default is 0.25 as recommended in the focal loss paper.
    - gamma (float, optional): Focusing parameter for focal loss. Higher values focus more on hard examples.
                               Default is 2.0 as recommended in the focal loss paper.
    - apply_sigmoid (bool, optional): Whether to apply the sigmoid activation to `y_pred`.
                                      Should be True when `y_pred` contains raw logits.
                                      Default is True for numerical stability.

    Returns:
    - float: A single scalar loss value (sum of per-pixel focal loss, following the focal loss paper).
    """


    # Validate inputs
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
        raise TypeError("y_pred and y_true must be numpy arrays.")
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must have the same shape.")
    
    # Validate mask shape if provided
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a numpy array.")
        if mask.shape != y_true.shape:
            raise ValueError("mask must have the same shape as y_true.")
    
    # Convert inputs to float32 for downstream calculations
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)

    # Apply sigmoid if the input is logits (common from semenatic segmentation models)
    if apply_sigmoid:
        y_pred = np.clip(y_pred, -20, 20)  # prevent overflow
        y_pred = 1 / (1 + np.exp(-y_pred))

    # Define p_t as defined in the paper
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true) # vectorized implementation
    p_t = np.clip(p_t, 1e-6, 1 - 1e-6) # clip p_t to prevent log(0)


    # Define alpha_t the same way as p_t
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)

    # Compute the Focal Loss
    focal_loss = -alpha_t * (1 - p_t) ** gamma * np.log(p_t + 1e-10)  # apply small epsilon to prevent log(0)

    # Apply mask if provided
    if mask is not None:
        return np.sum(focal_loss * mask.astype(np.float32))  # sum only valid pixels if mask provided
    else:
        return np.sum(focal_loss)  # sum all pixels
