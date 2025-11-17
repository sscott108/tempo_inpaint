import torch
import numpy as np
from sklearn.metrics import r2_score
import torch.nn.functional as F
import torchvision.transforms.functional as TF

EPS = 1e-8

# def calculate_metrics(pred, target, mask, known_mask, p_mask=None, p_values=None, normalizer=None):

#     metrics = {
#         'rmse': 0.0, 
#         'mae': 0.0, 
#         'r2': 0.0,
#         'pandora_rmse': 0.0,
#         'pandora_mae': 0.0,
#         'pandora_rho': 0.0,
#         'pandora_r2': 0.0,
#         'n_pandora_stations': 0
#     }
    
#     # For training: only consider artificial holes that were originally valid TEMPO pixels
#     # artificial_holes = pixels that are masked in current mask BUT were valid in known_mask
#     artificial_holes = (known_mask == 1) & (mask == 0)
    
#     if artificial_holes.sum() > 0:
#         pred_holes = pred[artificial_holes]
#         target_holes = target[artificial_holes]
        
#         # Calculate metrics
#         mse = torch.mean((pred_holes - target_holes) ** 2).item()
#         metrics['rmse'] = np.sqrt(mse)
#         metrics['mae'] = torch.mean(torch.abs(pred_holes - target_holes)).item()
        
#         # For R², convert to numpy
#         pred_np = pred_holes.detach().cpu().numpy()
#         target_np = target_holes.detach().cpu().numpy()
        
#         # Filter out NaN values
#         valid_idx = np.isfinite(pred_np) & np.isfinite(target_np)
#         if valid_idx.sum() > 1:
#             try:
#                 from sklearn.metrics import r2_score
#                 metrics['r2'] = r2_score(target_np[valid_idx], pred_np[valid_idx])
#             except Exception as e:
#                 print(f"Warning: Could not calculate R² for artificial holes: {e}")
#                 metrics['r2'] = 0.0
    
#     # If Pandora data is available, calculate Pandora metrics
#     if p_mask is not None and p_values is not None and p_mask.sum() > 0:
#         try:
#             # Handle tensor shapes - squeeze the channel dimension from predictions
#             pred_squeezed = pred.squeeze(1)  # [B, H, W]
            
#             # Get Pandora locations - p_mask should be [B, H, W]
#             pandora_locations = p_mask.bool()
            
#             # Get prediction values at Pandora locations
#             pred_pandora = pred_squeezed[pandora_locations]
            
#             # Get Pandora values
#             pandora_vals = p_values[pandora_locations]
            
#             # Count Pandora stations
#             n_stations = pandora_locations.sum().item()
#             metrics['n_pandora_stations'] = n_stations
            
#             if n_stations > 0:
#                 # Denormalize Pandora values if normalizer is provided
#                 if normalizer is not None:
#                     try:
#                         if hasattr(normalizer, 'denormalize_pandora'):
#                             # Handle individual denormalization
#                             pandora_vals_np = pandora_vals.detach().cpu().numpy()
#                             pandora_vals_denorm = np.array([normalizer.denormalize_pandora(v) 
#                                                         for v in pandora_vals_np])
#                         elif hasattr(normalizer, 'denormalize_pandora_array'):
#                             # Handle array denormalization
#                             pandora_vals_denorm = normalizer.denormalize_pandora_array(
#                                               pandora_vals.detach().cpu().numpy())
#                         else:
#                             pandora_vals_denorm = pandora_vals.detach().cpu().numpy()
#                     except Exception as e:
#                         print(f"Warning: Pandora denormalization failed: {e}")
#                         pandora_vals_denorm = pandora_vals.detach().cpu().numpy()
#                 else:
#                     pandora_vals_denorm = pandora_vals.detach().cpu().numpy()
                
#                 # Denormalize predictions at Pandora locations
#                 if normalizer is not None and hasattr(normalizer, 'denormalize_image'):
#                     try:
#                         # Detach and convert to numpy, then denormalize
#                         pred_pandora_np = pred_pandora.detach().cpu().numpy()
#                         pred_pandora_denorm = np.array([normalizer.denormalize_image(np.array([[p]]))[0,0] 
#                                                       for p in pred_pandora_np])
#                     except Exception as e:
#                         print(f"Warning: Prediction denormalization failed: {e}")
#                         pred_pandora_denorm = pred_pandora.detach().cpu().numpy()
#                 else:
#                     pred_pandora_denorm = pred_pandora.detach().cpu().numpy()
                
#                 # Filter out NaN values
#                 valid_mask = np.isfinite(pred_pandora_denorm) & np.isfinite(pandora_vals_denorm)
                
#                 if valid_mask.sum() > 0:
#                     pred_valid = pred_pandora_denorm[valid_mask]
#                     pandora_valid = pandora_vals_denorm[valid_mask]
                    
#                     # Calculate Pandora metrics
#                     metrics['pandora_rmse'] = np.sqrt(np.mean((pred_valid - pandora_valid) ** 2))
#                     metrics['pandora_mae'] = np.mean(np.abs(pred_valid - pandora_valid))
                    
#                     # For correlation and R², need at least 2 points
#                     if valid_mask.sum() > 1:
#                         # Calculate Spearman correlation
#                         try:
#                             from scipy import stats
#                             rho, _ = stats.spearmanr(pred_valid, pandora_valid)
#                             metrics['pandora_rho'] = float(rho) if np.isfinite(rho) else 0.0
#                         except Exception as e:
#                             print(f"Warning: Spearman correlation calculation failed: {e}")
#                             metrics['pandora_rho'] = 0.0
                        
#                         # Calculate R²
#                         try:
#                             from sklearn.metrics import r2_score
#                             metrics['pandora_r2'] = r2_score(pandora_valid, pred_valid)
#                         except Exception as e:
#                             print(f"Warning: Pandora R² calculation failed: {e}")
#                             metrics['pandora_r2'] = 0.0
        
#         except Exception as e:
#             print(f"Warning: Pandora metrics calculation failed: {e}")
    
#     return metrics
def calculate_metrics(pred, target, mask, known_mask, p_mask=None, p_values=None, normalizer=None):
    metrics = {
        'rmse': 0.0, 'mae': 0.0, 'r2': 0.0,
        'pandora_rmse': 0.0, 'pandora_mae': 0.0,
        'pandora_rho': 0.0, 'pandora_r2': 0.0,
        'n_pandora_stations': 0
    }

    # --- 1. Basic artificial-hole metrics
    artificial_holes = (known_mask == 1) & (mask == 0)
    if artificial_holes.sum() > 0:
        pred_holes = pred[artificial_holes]
        target_holes = target[artificial_holes]
        mse = torch.mean((pred_holes - target_holes) ** 2).item()
        metrics['rmse'] = np.sqrt(mse)
        metrics['mae'] = torch.mean(torch.abs(pred_holes - target_holes)).item()

        pred_np = pred_holes.detach().cpu().numpy()
        target_np = target_holes.detach().cpu().numpy()
        valid = np.isfinite(pred_np) & np.isfinite(target_np)
        if valid.sum() > 1:
            from sklearn.metrics import r2_score
            metrics['r2'] = r2_score(target_np[valid], pred_np[valid])

    # --- 2. Pandora metrics
    if p_mask is None or p_values is None:
        return metrics
    try:
        # Normalize shape: ensure [B,H,W]
        if p_mask.ndim == 2:
            p_mask = p_mask.unsqueeze(0)
        if p_mask.ndim == 4:
            p_mask = p_mask.squeeze(1)
        if p_values.ndim == 2:
            p_values = p_values.unsqueeze(0)
        if p_values.ndim == 4:
            p_values = p_values.squeeze(1)

        # Also squeeze predictions to [B,H,W]
        if pred.ndim == 4:
            pred_squeezed = pred.squeeze(1)
        else:
            pred_squeezed = pred

        # Handle possible batch mismatch (use first item if needed)
        if p_mask.shape[0] != pred_squeezed.shape[0]:
            p_mask = p_mask[0:1]
            p_values = p_values[0:1]
            pred_squeezed = pred_squeezed[0:1]

        pandora_locations = p_mask.bool()
        pred_pandora = pred_squeezed[pandora_locations]
        pandora_vals = p_values[pandora_locations]

        n_stations = pandora_locations.sum().item()
        metrics['n_pandora_stations'] = n_stations

        if n_stations > 0:
            # --- Denormalize Pandora and pred values
            if normalizer is not None:
                if hasattr(normalizer, "denormalize_pandora_array"):
                    pandora_vals_denorm = normalizer.denormalize_pandora_array(
                        pandora_vals.detach().cpu().numpy()
                    )
                elif hasattr(normalizer, "denormalize_pandora"):
                    pandora_vals_denorm = np.array([
                        normalizer.denormalize_pandora(v.item()) for v in pandora_vals
                    ])
                else:
                    pandora_vals_denorm = pandora_vals.detach().cpu().numpy()
            else:
                pandora_vals_denorm = pandora_vals.detach().cpu().numpy()

            if normalizer is not None and hasattr(normalizer, "denormalize_image"):
                pred_pandora_denorm = np.array([
                    normalizer.denormalize_image(np.array([[v.item()]]) )[0,0]
                    for v in pred_pandora
                ])
            else:
                pred_pandora_denorm = pred_pandora.detach().cpu().numpy()

            valid = np.isfinite(pred_pandora_denorm) & np.isfinite(pandora_vals_denorm)
            if valid.sum() > 0:
                pred_v = pred_pandora_denorm[valid]
                pandora_v = pandora_vals_denorm[valid]

                metrics['pandora_rmse'] = np.sqrt(np.mean((pred_v - pandora_v) ** 2))
                metrics['pandora_mae'] = np.mean(np.abs(pred_v - pandora_v))

                if valid.sum() > 1:
                    from scipy import stats
                    from sklearn.metrics import r2_score
                    rho, _ = stats.spearmanr(pred_v, pandora_v)
                    metrics['pandora_rho'] = float(rho) if np.isfinite(rho) else 0.0
                    metrics['pandora_r2'] = r2_score(pandora_v, pred_v)
    except Exception as e:
        print(f"Warning: Pandora metrics calculation failed: {e}")

    return metrics


def _dilate_mask(bin_mask, k=3):
    """bin_mask: [B,1,H,W], 0/1. Returns dilated binary mask."""
    pad = k // 2
    kernel = torch.ones(1, 1, k, k, device=bin_mask.device, dtype=bin_mask.dtype)
    hits = F.conv2d(bin_mask, kernel, padding=pad)
    return (hits > 0).float()

def _finite_diff_xy(x):
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    return dy, dx

def boundary_ring_loss(pred, target, hole_mask, ring_width=5, w_val=1.0, w_grad=1.0, eps=1e-8):
    """
    pred/target: [B,1,H,W], hole_mask: [B,1,H,W] with 1=hole, 0=known.
    """
    hole = (hole_mask > 0.5).float()
    ring = _dilate_mask(hole, k=ring_width) - hole
    ring = (ring > 0.5).float()  # [B,1,H,W]

    # value continuity on ring
    val_err = torch.abs(pred - target) * ring
    val_term = val_err.sum() / (ring.sum() + eps)

    # gradient continuity on ring (align overlapping pixels)
    dy_p, dx_p = _finite_diff_xy(pred)
    dy_t, dx_t = _finite_diff_xy(target)

    ring_y = ring[:, :, 1:, :] * ring[:, :, :-1, :]
    ring_x = ring[:, :, :, 1:] * ring[:, :, :, :-1]

    gy = torch.abs(dy_p - dy_t) * ring_y
    gx = torch.abs(dx_p - dx_t) * ring_x
    grad_term = (gy.sum() + gx.sum()) / (ring_y.sum() + ring_x.sum() + eps)

    return w_val * val_term + w_grad * grad_term

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

def highfreq_loss(pred, target, sigma=6):
    # Kernel size should be odd and ≈ 6*sigma
    ksize = int(2 * round(3 * sigma) + 1)

    # Apply Gaussian blur
    pred_lp = TF.gaussian_blur(pred, kernel_size=ksize, sigma=sigma)
    target_lp = TF.gaussian_blur(target, kernel_size=ksize, sigma=sigma)

    # High-pass residuals
    pred_hp = pred - pred_lp
    target_hp = target - target_lp

    return F.l1_loss(pred_hp, target_hp)

def improved_loss_progress(pred, target, mask, epoch, max_epochs):
    hole_mask = (mask == 0).float()

    # --- Core losses ---
    l1_loss = F.l1_loss(pred * hole_mask, target * hole_mask)
    l2_loss = F.mse_loss(pred * hole_mask, target * hole_mask)   # <-- new L2 loss
    grad_loss = gradient_loss(pred, target, mask)
    tv_loss   = total_variation_loss(pred, mask)
    L_ring    = boundary_ring_loss(pred, target, hole_mask, ring_width=3)

    # --- Multi-scale high-frequency loss ---
    loss_hf = (
        1.0  * highfreq_loss(pred, target, sigma=1) +
        0.5  * highfreq_loss(pred, target, sigma=2) +
        0.25 * highfreq_loss(pred, target, sigma=4)
    )

    # --- Progress factor ---
    frac = min(epoch / max_epochs, 1.0)

    # --- Stage weighting ---
    w_l1   = 1.0 - 0.3*frac          # decay L1 over time
    w_l2   = 0.3 + 0.2*frac          # grow L2 as predictions stabilize
    w_grad = 0.05 + 0.05*frac
    w_tv   = 0.02 + 0.05*frac
    w_ring = 0.0 + 0.2*frac
    w_hf   = 0.1 + 0.2*frac

    # --- Combine ---
    total_loss = (
        w_l1   * l1_loss +
        w_l2   * l2_loss +       # <-- new term added here
        w_grad * grad_loss +
        w_tv   * tv_loss +
        w_ring * L_ring +
        w_hf   * loss_hf
    )

    return total_loss


