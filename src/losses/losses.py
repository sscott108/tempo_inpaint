import torch
import numpy as np
from sklearn.metrics import r2_score
import torch.nn.functional as F
import torchvision.transforms.functional as TF

EPS = 1e-8

def calculate_metrics(pred, target, mask, known_mask, p_mask=None, p_values=None, normalizer=None):

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

def improved_loss_progress(pred, target, mask, epoch, max_epochs):
    hole_mask = (mask == 0).float()

    # Core losses
    l1_loss = F.l1_loss(pred * hole_mask, target * hole_mask)
    grad_loss = gradient_loss(pred, target, mask)
    tv_loss   = total_variation_loss(pred, mask)
    L_ring    = boundary_ring_loss(pred, target, hole_mask, ring_width=3)

    # Multi-scale high-frequency loss
    loss_hf = (
        1.0 * highfreq_loss(pred, target, sigma=1) +
        0.5 * highfreq_loss(pred, target, sigma=2) +
        0.25 * highfreq_loss(pred, target, sigma=4)
    )

    # Progress factor
    frac = min(epoch / max_epochs, 1.0)

    # Stage weighting
    # Early: reconstruction-heavy
    # Mid: boundary + smoothness
    # Late: details & texture dominate
    w_l1   = 1.0 - 0.3*frac          # decay L1 over time
    w_grad = 0.05 + 0.05*frac        # small, steady
    w_tv   = 0.02 + 0.05*frac        # small, steady
    w_ring = 0.0 + 0.2*frac          # kicks in mid → late
    w_hf   = 0.1 + 0.2*frac          # grows with training

    total_loss = (
        w_l1   * l1_loss +
        w_grad * grad_loss +
        w_tv   * tv_loss +
        w_ring * L_ring +
        w_hf   * loss_hf
    )

    return total_loss


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

def improved_loss_progress(pred, target, mask, epoch, max_epochs, d_fake=None):
    """
    pred: model output
    target: ground truth
    mask: binary keep mask (1=keep, 0=hole)
    epoch, max_epochs: for scheduling
    d_fake: optional discriminator output for adversarial term
    """
    hole_mask  = (mask == 0).float()
    valid_mask = (mask > 0).float()

    # Pixelwise reconstruction losses
    l1_hole  = F.l1_loss(pred * hole_mask,  target * hole_mask)
    l1_valid = F.l1_loss(pred * valid_mask, target * valid_mask)

    # Core regularization terms
    grad_loss = gradient_loss(pred, target, mask)
    tv_loss   = total_variation_loss(pred, mask)
    L_ring    = boundary_ring_loss(pred, target, hole_mask, ring_width=3)

    # Multi-scale high-frequency loss
    loss_hf = (
        1.0  * highfreq_loss(pred, target, sigma=1) +
        0.5  * highfreq_loss(pred, target, sigma=2) +
        0.25 * highfreq_loss(pred, target, sigma=4)
    )

    # Training progression (0 → 1)
    frac = min(epoch / max_epochs, 1.0)

    # ---------------------------
    # Phase-based scheduling
    # ---------------------------
    # Phase 1: Pure reconstruction (0–20%)
    # Phase 2: Introduce texture + boundary + adversarial (20–60%)
    # Phase 3: Texture + adversarial dominate (60–100%)

    if frac < 0.2:  # Phase 1
        w_hole, w_valid = 1.0, 0.1
        w_grad, w_tv, w_ring, w_hf, w_adv = 0.05, 0.05, 0.0, 0.0, 0.0

    elif frac < 0.6:  # Phase 2
        progress = (frac - 0.2) / 0.4  # 0 → 1 across 20–60%
        w_hole  = 1.0 - 0.2 * progress
        w_valid = 0.1 + 0.3 * progress   # grows to ~0.4
        w_grad  = 0.05 + 0.05 * progress
        w_tv    = 0.05 * (1 - progress)  # decays
        w_ring  = 0.2 * progress
        w_hf    = 0.1 + 0.2 * progress
        w_adv   = 0.0 + 0.3 * progress   # delayed ramp

    else:  # Phase 3
        progress = (frac - 0.6) / 0.4  # 0 → 1 across 60–100%
        w_hole  = 0.8 - 0.2 * progress
        w_valid = 0.4 + 0.1 * progress   # ends ~0.5
        w_grad  = 0.1
        w_tv    = 0.02 * (1 - progress)  # fades out
        w_ring  = 0.2 + 0.2 * progress   # up to 0.4
        w_hf    = 0.3 + 0.2 * progress   # up to 0.5
        w_adv   = 0.3 + 0.1 * progress   # up to 0.4

    # Reconstruction loss (with dynamic valid weighting)
    l1_loss = w_hole * l1_hole + w_valid * l1_valid

    # Total generator loss
    total_loss = (
        l1_loss +
        w_grad * grad_loss +
        w_tv   * tv_loss +
        w_ring * L_ring +
        w_hf   * loss_hf
    )

    # Add adversarial term if discriminator is active
    if d_fake is not None and w_adv > 0:
        loss_G_adv = g_adv_loss(d_fake)
        total_loss += w_adv * loss_G_adv

    return total_loss


def d_loss(d_real, d_fake): return 0.5 * (torch.mean((d_real - 1)**2) + torch.mean(d_fake**2))
def g_adv_loss(d_fake): return torch.mean((d_fake - 1)**2)
