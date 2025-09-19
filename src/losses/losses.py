import torch
import numpy as np
from sklearn.metrics import r2_score
import torch.nn.functional as F

def calculate_metrics(pred, target, mask, known_mask, p_mask=None, p_values=None, normalizer=None):
    """
    Calculate RMSE, MAE, R² for artificially masked regions and Pandora stations
    
    Args:
        pred: Model predictions [B,C,H,W]
        target: Ground truth target [B,C,H,W]
        mask: Current mask being used (fake_mask for training, known_mask for validation) [B,C,H,W]
        known_mask: Original TEMPO valid pixel mask (1=valid, 0=missing) [B,C,H,W]
        p_mask: Pandora mask (1=station location, 0=no station) [B,H,W] or None
        p_values: Normalized Pandora values at station locations [B,H,W] or None
        normalizer: Normalizer class with denormalize_pandora or denormalize_pandora_array methods
    
    Returns:
        dict: Dictionary with RMSE, MAE, R² for artificial holes and Pandora stations
    """
    # Initialize results dictionary
    metrics = {
        'rmse': 0.0, 
        'mae': 0.0, 
        'r2': 0.0,
        'pandora_rmse': 0.0,
        'pandora_mae': 0.0,
        'pandora_rho': 0.0,
        'pandora_r2': 0.0,
        'n_pandora_stations': 0
    }
    
    # For training: only consider artificial holes that were originally valid TEMPO pixels
    # artificial_holes = pixels that are masked in current mask BUT were valid in known_mask
    artificial_holes = (known_mask == 1) & (mask == 0)
    
    if artificial_holes.sum() > 0:
        pred_holes = pred[artificial_holes]
        target_holes = target[artificial_holes]
        
        # Calculate metrics
        mse = torch.mean((pred_holes - target_holes) ** 2).item()
        metrics['rmse'] = np.sqrt(mse)
        metrics['mae'] = torch.mean(torch.abs(pred_holes - target_holes)).item()
        
        # For R², convert to numpy
        pred_np = pred_holes.detach().cpu().numpy()
        target_np = target_holes.detach().cpu().numpy()
        
        # Filter out NaN values
        valid_idx = np.isfinite(pred_np) & np.isfinite(target_np)
        if valid_idx.sum() > 1:
            try:
                from sklearn.metrics import r2_score
                metrics['r2'] = r2_score(target_np[valid_idx], pred_np[valid_idx])
            except Exception as e:
                print(f"Warning: Could not calculate R² for artificial holes: {e}")
                metrics['r2'] = 0.0
    
    # If Pandora data is available, calculate Pandora metrics
    if p_mask is not None and p_values is not None and p_mask.sum() > 0:
        try:
            # Handle tensor shapes - squeeze the channel dimension from predictions
            pred_squeezed = pred.squeeze(1)  # [B, H, W]
            
            # Get Pandora locations - p_mask should be [B, H, W]
            pandora_locations = p_mask.bool()
            
            # Get prediction values at Pandora locations
            pred_pandora = pred_squeezed[pandora_locations]
            
            # Get Pandora values
            pandora_vals = p_values[pandora_locations]
            
            # Count Pandora stations
            n_stations = pandora_locations.sum().item()
            metrics['n_pandora_stations'] = n_stations
            
            if n_stations > 0:
                # Denormalize Pandora values if normalizer is provided
                if normalizer is not None:
                    try:
                        if hasattr(normalizer, 'denormalize_pandora'):
                            # Handle individual denormalization
                            pandora_vals_np = pandora_vals.detach().cpu().numpy()
                            pandora_vals_denorm = np.array([normalizer.denormalize_pandora(v) 
                                                        for v in pandora_vals_np])
                        elif hasattr(normalizer, 'denormalize_pandora_array'):
                            # Handle array denormalization
                            pandora_vals_denorm = normalizer.denormalize_pandora_array(
                                              pandora_vals.detach().cpu().numpy())
                        else:
                            pandora_vals_denorm = pandora_vals.detach().cpu().numpy()
                    except Exception as e:
                        print(f"Warning: Pandora denormalization failed: {e}")
                        pandora_vals_denorm = pandora_vals.detach().cpu().numpy()
                else:
                    pandora_vals_denorm = pandora_vals.detach().cpu().numpy()
                
                # Denormalize predictions at Pandora locations
                if normalizer is not None and hasattr(normalizer, 'denormalize_image'):
                    try:
                        # Detach and convert to numpy, then denormalize
                        pred_pandora_np = pred_pandora.detach().cpu().numpy()
                        pred_pandora_denorm = np.array([normalizer.denormalize_image(np.array([[p]]))[0,0] 
                                                      for p in pred_pandora_np])
                    except Exception as e:
                        print(f"Warning: Prediction denormalization failed: {e}")
                        pred_pandora_denorm = pred_pandora.detach().cpu().numpy()
                else:
                    pred_pandora_denorm = pred_pandora.detach().cpu().numpy()
                
                # Filter out NaN values
                valid_mask = np.isfinite(pred_pandora_denorm) & np.isfinite(pandora_vals_denorm)
                
                if valid_mask.sum() > 0:
                    pred_valid = pred_pandora_denorm[valid_mask]
                    pandora_valid = pandora_vals_denorm[valid_mask]
                    
                    # Calculate Pandora metrics
                    metrics['pandora_rmse'] = np.sqrt(np.mean((pred_valid - pandora_valid) ** 2))
                    metrics['pandora_mae'] = np.mean(np.abs(pred_valid - pandora_valid))
                    
                    # For correlation and R², need at least 2 points
                    if valid_mask.sum() > 1:
                        # Calculate Spearman correlation
                        try:
                            from scipy import stats
                            rho, _ = stats.spearmanr(pred_valid, pandora_valid)
                            metrics['pandora_rho'] = float(rho) if np.isfinite(rho) else 0.0
                        except Exception as e:
                            print(f"Warning: Spearman correlation calculation failed: {e}")
                            metrics['pandora_rho'] = 0.0
                        
                        # Calculate R²
                        try:
                            from sklearn.metrics import r2_score
                            metrics['pandora_r2'] = r2_score(pandora_valid, pred_valid)
                        except Exception as e:
                            print(f"Warning: Pandora R² calculation failed: {e}")
                            metrics['pandora_r2'] = 0.0
        
        except Exception as e:
            print(f"Warning: Pandora metrics calculation failed: {e}")
    
    return metrics

def gradient_loss(pred, target, mask=None):
    """Penalize large gradients to encourage smoothness"""
    # Calculate gradients
    dy_pred, dx_pred = torch.gradient(pred, dim=(-2, -1))
    dy_target, dx_target = torch.gradient(target, dim=(-2, -1))
    
    if mask is not None:
        # Only calculate loss in valid regions
        mask_expanded = mask.expand_as(dy_pred)
        dy_pred = dy_pred * mask_expanded
        dx_pred = dx_pred * mask_expanded
        dy_target = dy_target * mask_expanded
        dx_target = dx_target * mask_expanded
    
    # L1 loss on gradients
    grad_loss = F.l1_loss(dy_pred, dy_target) + F.l1_loss(dx_pred, dx_target)
    return grad_loss

def total_variation_loss(pred, mask=None):
    """Total variation loss for smoothness"""
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    
    if mask is not None:
        # Apply mask to gradients
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        dy = dy * mask_y
        dx = dx * mask_x
    
    return torch.mean(dy) + torch.mean(dx)

# Updated loss function
def improved_loss(pred, target, mask):
    # Main reconstruction loss
    hole_mask = (mask == 0).float()
    l1_loss = F.l1_loss(pred * hole_mask, target * hole_mask)
    
    # Smoothness losses
    grad_loss = gradient_loss(pred, target, mask)
    tv_loss = total_variation_loss(pred, mask)
    
    # Combine losses
    total_loss = l1_loss + 0.1 * grad_loss + 0.05 * tv_loss
    return total_loss