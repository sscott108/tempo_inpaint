import torch
import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(pred, target, mask, known_mask):
    """
    Calculate RMSE, MAE, R² only for artificially masked regions that had valid TEMPO pixels
    
    Args:
        pred: Model predictions
        target: Ground truth target
        mask: Current mask being used (fake_mask for training, known_mask for validation)
        known_mask: Original TEMPO valid pixel mask (1=valid, 0=missing)
    """
    # For training: only consider artificial holes that were originally valid TEMPO pixels
    # artificial_holes = pixels that are masked in current mask BUT were valid in known_mask
    artificial_holes = (known_mask == 1) & (mask == 0)
    
    if artificial_holes.sum() == 0:
        return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}  # No artificial holes in valid TEMPO areas
        
    pred_holes = pred[artificial_holes]
    target_holes = target[artificial_holes]
    
    # Calculate metrics
    mse = torch.mean((pred_holes - target_holes) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(pred_holes - target_holes)).item()
    
    # For R², convert to numpy
    pred_np = pred_holes.detach().cpu().numpy()
    target_np = target_holes.detach().cpu().numpy()
    r2 = r2_score(target_np, pred_np) if len(pred_np) > 1 else 0.0
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def warmup_loss(pred, target, mask):
    hole_mask = 1- mask
    if mask.ndim == 3:  # ensure channel dim
        mask = mask.unsqueeze(1)
    valid_mask = mask.float()

    diff = (pred - target) * valid_mask
    l1_iden = torch.sum(torch.abs(diff)) / (valid_mask.sum() + 1e-8)

    return l1_iden 