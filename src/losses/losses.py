import torch
import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(pred, target, mask):
    """Calculate RMSE, MAE, R² for the fake mask regions (where mask=0)"""
    # Only consider the fake mask regions (where mask=0)
    hole_mask = (1 - mask).bool()
    
    # Extract values in the fake mask regions
    if hole_mask.sum() == 0:
        return {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}  # No fake mask regions
        
    pred_holes = pred[hole_mask]
    target_holes = target[hole_mask]
    
    # Calculate metrics
    mse = torch.mean((pred_holes - target_holes) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(pred_holes - target_holes)).item()
    
    # For R², convert to numpy for sklearn implementation
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