import torch
import numpy as np
from sklearn.metrics import r2_score

def calculate_metrics(pred, target, mask, known_mask, p_mask=None, p_values=None, normalizer=None, batch_idx=0):
    """
    Calculate metrics for artificial holes and Pandora stations
    """
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
    
    try:
        # Ensure batch_idx is within range
        if batch_idx >= pred.shape[0]:
            print(f"Warning: batch_idx {batch_idx} is out of range for tensor with {pred.shape[0]} batches")
            return metrics
        
        # Extract single batch elements to avoid batch size issues
        pred_single = pred[batch_idx:batch_idx+1]
        target_single = target[batch_idx:batch_idx+1]
        mask_single = mask[batch_idx:batch_idx+1]
        known_mask_single = known_mask[batch_idx:batch_idx+1]
        
        # For Pandora data, check if available
        p_mask_single = None
        p_values_single = None
        if p_mask is not None and p_values is not None:
            p_mask_single = p_mask[batch_idx:batch_idx+1]
            p_values_single = p_values[batch_idx:batch_idx+1]
        
        # Now process with single-batch tensors
        artificial_holes = (known_mask_single == 1) & (mask_single == 0)
        
        if artificial_holes.sum() > 0:
            # Use single_batch tensors for indexing too - THIS WAS THE BUG
            pred_holes = pred_single[artificial_holes]
            target_holes = target_single[artificial_holes]
            
            # Calculate metrics only if we have valid data
            if len(pred_holes) > 0:
                mse = torch.mean((pred_holes - target_holes) ** 2).item()
                metrics['rmse'] = np.sqrt(mse) if np.isfinite(mse) else 0.0
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
        if p_mask_single is not None and p_values_single is not None:
            try:
                # Handle tensor shapes - squeeze the channel dimension from predictions
                if len(pred_single.shape) == 4 and pred_single.shape[1] == 1:
                    pred_squeezed = pred_single.squeeze(1)  # [1, H, W]
                elif len(pred_single.shape) == 3:
                    pred_squeezed = pred_single  # Already [1, H, W]
                else:
                    print(f"Warning: Unexpected prediction shape: {pred_single.shape}")
                    return metrics
                
                # Ensure p_mask is boolean
                pandora_locations = p_mask_single.bool()
                
                # Get prediction values at Pandora locations
                if pandora_locations.sum() > 0:
                    pred_pandora = pred_squeezed[pandora_locations]
                    pandora_vals = p_values_single[pandora_locations]
                    
                    # Count Pandora stations
                    n_stations = pandora_locations.sum().item()
                    metrics['n_pandora_stations'] = n_stations
                    
                    if n_stations > 0:
                        # Denormalize Pandora values if normalizer is provided
                        if normalizer is not None:
                            try:
                                if hasattr(normalizer, 'denormalize_pandora'):
                                    pandora_vals_np = pandora_vals.detach().cpu().numpy()
                                    pandora_vals_denorm = np.array([normalizer.denormalize_pandora(float(v)) 
                                                                for v in pandora_vals_np])
                                elif hasattr(normalizer, 'denormalize_pandora_array'):
                                    pandora_vals_np = pandora_vals.detach().cpu().numpy()
                                    pandora_vals_denorm = normalizer.denormalize_pandora_array(pandora_vals_np)
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
                                pred_pandora_np = pred_pandora.detach().cpu().numpy()
                                pred_pandora_denorm = np.array([normalizer.denormalize_image(np.array([[float(p)]]))[0,0] 
                                                              for p in pred_pandora_np])
                            except Exception as e:
                                print(f"Warning: Prediction denormalization failed: {e}")
                                pred_pandora_denorm = pred_pandora.detach().cpu().numpy()
                        else:
                            pred_pandora_denorm = pred_pandora.detach().cpu().numpy()
                        
                        # Ensure both arrays are numpy arrays and finite
                        pred_pandora_denorm = np.array(pred_pandora_denorm, dtype=np.float32)
                        pandora_vals_denorm = np.array(pandora_vals_denorm, dtype=np.float32)
                        
                        # Filter out NaN values
                        valid_mask = np.isfinite(pred_pandora_denorm) & np.isfinite(pandora_vals_denorm)
                        
                        if valid_mask.sum() > 0:
                            pred_valid = pred_pandora_denorm[valid_mask]
                            pandora_valid = pandora_vals_denorm[valid_mask]
                            
                            # Calculate Pandora metrics
                            mse_pandora = np.mean((pred_valid - pandora_valid) ** 2)
                            metrics['pandora_rmse'] = np.sqrt(mse_pandora) if np.isfinite(mse_pandora) else 0.0
                            metrics['pandora_mae'] = np.mean(np.abs(pred_valid - pandora_valid))
                            
                            # For correlation and R², need at least 2 points
                            if valid_mask.sum() > 1:
                                # Calculate Spearman correlation
                                try:
                                    from scipy import stats
                                    if len(np.unique(pred_valid)) > 1 and len(np.unique(pandora_valid)) > 1:
                                        rho, _ = stats.spearmanr(pred_valid, pandora_valid)
                                        metrics['pandora_rho'] = float(rho) if np.isfinite(rho) else 0.0
                                except Exception as e:
                                    print(f"Warning: Spearman correlation calculation failed: {e}")
                                
                                # Calculate R²
                                try:
                                    from sklearn.metrics import r2_score
                                    if len(np.unique(pandora_valid)) > 1:
                                        metrics['pandora_r2'] = r2_score(pandora_valid, pred_valid)
                                except Exception as e:
                                    print(f"Warning: Pandora R² calculation failed: {e}")
            except Exception as e:
                print(f"Warning: Pandora metrics calculation failed: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return metrics
def warmup_loss(pred, target, mask):
    hole_mask = 1- mask
    if mask.ndim == 3:  # ensure channel dim
        mask = mask.unsqueeze(1)
    valid_mask = mask.float()

    diff = (pred - target) * valid_mask
    l1_iden = torch.sum(torch.abs(diff)) / (valid_mask.sum() + 1e-8)

    return l1_iden 