import os
import re
import numpy as np
import shapefile
from datetime import datetime
from pyproj import Transformer
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi
import rasterio
from scipy.ndimage import gaussian_filter
import torch
from pyproj import CRS, Transformer
from scipy import stats

def visualize_batch(epoch, model, normalizer, dataloader, batch_idx=0, sample_idx=0, device="cuda", save=False, train=True, shp_path=None, avg_thr=0.2):
    """
    Visualize model predictions from a DataLoader
    
    Args:
        epoch: Current epoch number
        model: The model to evaluate
        normalizer: Normalizer for denormalizing images
        dataloader: DataLoader (either train or validation)
        batch_idx: Which batch to use from the dataloader (default: 0)
        sample_idx: Which sample within the batch to visualize (default: 0)
        device: Device to run inference on
        save: Whether to save the plot
        train: Whether this is training data (affects what gets plotted)
        shp_path: Path to shapefile for overlay
        avg_thr: Threshold for texture analysis
    """
    model.eval()
    
    # Get a batch from the dataloader
    try:
        # Get the specified batch
        for i, batch in enumerate(dataloader):
            if i == batch_idx:
                break
        else:
            print(f"Batch index {batch_idx} not found in dataloader")
            return
    except Exception as e:
        print(f"Error getting batch from dataloader: {e}")
        return
    
    # Extract the specific sample from the batch
    batch_size = batch["masked_img"].shape[0]
    if sample_idx >= batch_size:
        print(f"Sample index {sample_idx} not available in batch of size {batch_size}")
        sample_idx = 0
        print(f"Using sample index {sample_idx} instead")
    
    # Create sample dict by extracting the specific index from each tensor
    sample = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sample[key] = value[sample_idx]  # Extract specific sample
        elif isinstance(value, (list, tuple)):
            sample[key] = value[sample_idx]  # For paths or other list items
        else:
            sample[key] = value  # For single values

#     print(f"Visualizing batch {batch_idx}, sample {sample_idx} from dataloader")
    
    # Get common data available in both train and val
    img = sample["masked_img"].unsqueeze(0).to(device)   # [1,1,H,W]
    mask_obs = sample["known_mask"].unsqueeze(0).to(device)  # real missing tempo pixels
    target = sample["target"].unsqueeze(0).to(device)   # [1,1,H,W]
    
    # For training, get additional masks
    if train:
        mask = sample["known_and_fake_mask"].unsqueeze(0).to(device)  # [1,1,H,W]
        fake_mask = sample['fake_mask'].unsqueeze(0).to(device)
        fake_mask_np = fake_mask[0,0].cpu().numpy().astype(bool)
    else:
        # For validation, use only the known mask for inference
        mask = mask_obs
    
    # Get model prediction
    with torch.no_grad():
        pred, out_mask = model(img, mask)
    
    # Convert to numpy
    inp_np = normalizer.denormalize_image(img[0,0].cpu().numpy())
    mask_obs_np = mask_obs[0,0].cpu().numpy().astype(bool)   # 1=known
    pred_np = normalizer.denormalize_image(pred[0,0].cpu().numpy())
    tgt_np = normalizer.denormalize_image(target[0,0].cpu().numpy())
    
    if train:
        mask_np = sample["known_and_fake_mask"][0].cpu().numpy().astype(bool)
    else:
        mask_np = mask_obs_np  # For validation, use only observed mask

    # Extract metadata
    path = sample["path"]
    base = os.path.basename(path)
    ts = re.sub(r"\D", "", base)[:14]
    date = datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    
    # Load shapefile segments if provided
    if shp_path is not None:
        try:
            with rasterio.open(path) as src:
                tr = src.transform
                crs = src.crs
                H, W = src.height, src.width

            segments = load_shapefile_segments_pyshp(shp_path, crs)
            if segments:
                inv = ~tr
                seg_pix = []
                for seg in segments:
                    cols, rows = [], []
                    for x, y in seg:
                        c, r = inv * (x, y)
                        cols.append(c)
                        rows.append(r)
                    seg_pix.append(np.column_stack([cols, rows]))
                segments = seg_pix

            def _add_shape(a, alpha=1.0, color="k"):
                a.add_collection(LineCollection(segments, colors=color, linewidths=0.6, zorder=6, alpha=alpha))
        except Exception as e:
            print(f"Warning: Could not load shapefile: {e}")
            def _add_shape(a, alpha=1.0, color="k"):
                pass
    else:
        def _add_shape(a, alpha=1.0, color="k"):
            pass  # Do nothing if no shapefile

    # Calculate performance metrics
    holes_total = (~mask_np).sum()
    holes_filled_valid = ((~mask_np) & np.isfinite(pred_np)).sum()
    fill_percentage = 100.0 * holes_filled_valid / max(holes_total, 1)

    # Calculate range consistency metric
    holes_mask = ~mask_np
    known_values = pred_np[mask_np]
    filled_values = pred_np[holes_mask]
    
    range_consistency = 0.0
    if len(known_values) > 0 and len(filled_values) > 0:
        # Filter out NaN values
        known_values_clean = known_values[np.isfinite(known_values)]
        filled_values_clean = filled_values[np.isfinite(filled_values)]
        
        if len(known_values_clean) > 0 and len(filled_values_clean) > 0:
            known_range = np.percentile(known_values_clean, [5, 95])
            filled_in_range = np.sum((filled_values_clean >= known_range[0]) & 
                                   (filled_values_clean <= known_range[1]))
            range_consistency = 100.0 * filled_in_range / len(filled_values_clean)

    # Set up colormap and scaling
    cmap_v = plt.cm.viridis.copy()
    cmap_v.set_bad(color="white")
    finite = np.isfinite(tgt_np)
    vmin, vmax = (np.percentile(tgt_np[finite], [2, 98]) if finite.any() else (0, 1))
    
    if train:
        sample_type = "Training"
        
        # Calculate gap statistics and RMSE for training
        artificial_holes = mask_obs_np & (~fake_mask_np)  # Areas that were known but made into gaps
        original_holes = (~mask_obs_np).sum()
        artificial_gaps = artificial_holes.sum()
        total_known_pixels = mask_obs_np.sum()
        current_mask_np = mask_np 
        gap_percentage = 100.0 * artificial_gaps / max(total_known_pixels, 1)
        
        known_mask_np = mask_obs_np  # Original TEMPO valid pixels
        current_mask_np = mask_np    # Current mask (includes artificial holes)

        # Artificial holes = areas that are valid in TEMPO but masked in current processing
        artificial_holes = known_mask_np & (~current_mask_np)

        if artificial_holes.sum() > 0:
            pred_in_artificial = pred_np[artificial_holes]
            target_in_artificial = tgt_np[artificial_holes]

            # Remove any NaN values for RMSE calculation
            valid_mask = np.isfinite(pred_in_artificial) & np.isfinite(target_in_artificial)
            if valid_mask.sum() > 0:
                pred_valid = pred_in_artificial[valid_mask]
                target_valid = target_in_artificial[valid_mask]

                rmse = np.sqrt(np.mean((pred_valid - target_valid)**2))

                reconstruction_title = f"Reconstruction {fill_percentage:.1f}% filled, RMSE: {rmse:.3f}"

                
                # Convert RMSE to a percentage accuracy score
                target_range = np.max(tgt_np[np.isfinite(tgt_np)]) - np.min(tgt_np[np.isfinite(tgt_np)])
                accuracy_percentage = 100.0 * (1 - rmse / max(target_range, 1e-6))
                accuracy_percentage = max(0, min(100, accuracy_percentage))
                rho,_ = stats.spearmanr(pred_valid, target_valid)
                reconstruction_title = f"Reconstruction\nRMSE: {rmse:.4E}|  $\\rho$: {rho:.2f}"
   
        
        fig, ax = plt.subplots(1, 5, figsize=(20, 4))
        
        # Panel 0: Input with all holes (white)
        im0 = ax[0].imshow(np.ma.array(inp_np, mask=~mask_np), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[0].set_title("Input (N/A = white)")
        _add_shape(ax[0], color="k")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        # Panel 1: Artificial mask (fake holes) - WITH GAP PERCENTAGE
        im1 = ax[1].imshow(~artificial_holes, cmap="Reds", alpha=0.8)
        ax[1].set_title(f"Artificial Holes ({gap_percentage:.1f}%)")
        _add_shape(ax[1], color="k")
        
        # Panel 2: Reconstruction - WITH RMSE
        im2 = ax[2].imshow(np.ma.array(pred_np, mask=~np.isfinite(pred_np)), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[2].set_title(reconstruction_title)
        _add_shape(ax[2], color="k")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        # Panel 3: Filled values only in holes - WITH RANGE CONSISTENCY
        filled_only = np.full_like(pred_np, np.nan, dtype=np.float32)
        filled_only[~mask_np] = pred_np[~mask_np]   # keep values only where mask==0
        im3 = ax[3].imshow(np.ma.array(filled_only, mask=np.isnan(filled_only)), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[3].set_title(f"Filled Values in Holes")
        _add_shape(ax[3], color="k")
        fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        # Panel 4: Target with original holes
        im4 = ax[4].imshow(np.ma.array(tgt_np, mask=~mask_obs_np), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[4].set_title("Target (original holes = white)")
        _add_shape(ax[4], color="k")
        fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
        
    else:
        sample_type = "Validation"
        
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        
        # Panel 0: Input with missing pixels as white
        im0 = ax[0].imshow(np.ma.array(inp_np, mask=~mask_obs_np), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[0].set_title("Input (N/A = white)")
        _add_shape(ax[0], color="k")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        
        # Panel 1: Reconstruction
        im1 = ax[1].imshow(np.ma.array(pred_np, mask=~np.isfinite(pred_np)), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[1].set_title("Reconstruction")
        _add_shape(ax[1], color="k")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        
        # Panel 2: Filled values only in holes - WITH RANGE CONSISTENCY
        filled_only = np.full_like(pred_np, np.nan, dtype=np.float32)
        filled_only[~mask_obs_np] = pred_np[~mask_obs_np]   # keep values only where mask==0
        im2 = ax[2].imshow(np.ma.array(filled_only, mask=np.isnan(filled_only)), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[2].set_title(f"Filled Values in Holes")
        _add_shape(ax[2], color="k")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        # Panel 3: Original target image (complete)
        im3 = ax[3].imshow(np.ma.array(tgt_np, mask=~np.isfinite(tgt_np)), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[3].set_title("Target Image")
        _add_shape(ax[3], color="k")
        fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    for a in ax: 
        a.axis("off")

    fig.suptitle(f"{sample_type} image {date}, epoch {epoch}", fontsize=16, y= 1.1)
    plt.tight_layout()
    
    save_path = path.split('/')[-1].split('.')[0]
    if save:
        plt.savefig(f'{save_path}_{sample_type}_epoch_{epoch}.png', dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    
def _add_shape(ax, segments, alpha=1.0, color="k"):
    """Helper function to add shapefile overlay to plot"""
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6, zorder=6, alpha=alpha))
    
def load_shapefile_segments_pyshp(shp_path, target_crs, src_epsg = 4326):
    """
    Load shapefile segments and transform to target CRS
    """
    src_crs = CRS.from_epsg(src_epsg)  # You'll need to determine source CRS
    tgt = target_crs if hasattr(target_crs, "to_wkt") else None
    if tgt is None:
        return None
    
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

    r = shapefile.Reader(shp_path)
    segments = []
    for shapeRec in r.shapeRecords():
        shp = shapeRec.shape
        pts = np.asarray(shp.points, dtype=float)
        if pts.size == 0: 
            continue
        xs, ys = transformer.transform(pts[:,0], pts[:,1])
        pts_t = np.column_stack([xs, ys])
        parts = list(shp.parts) + [len(pts_t)]
        for i in range(len(parts)-1):
            seg = pts_t[parts[i]:parts[i+1]]
            if seg.shape[0] >= 2:
                segments.append(seg)
    return segments

def extract_timestamp_from_path(path):
    """
    Extract timestamp from file path
    """
    base = os.path.basename(path)
    ts = re.sub(r"\D", "", base)[:14]
    date = datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    return date


def generate_realistic_gaps_simple(shape, tempo_mask, n_blobs=5, blob_size_range=(80, 200), threshold=0.5):
    """
    Generate realistic cloud-like gap patterns using multi-scale blobs
    
    Parameters:
    - shape: (ny, nx) shape of the output
    - tempo_mask: boolean array where True indicates valid TEMPO pixels
    - n_blobs: number of large-scale cloud structures
    - blob_size_range: tuple of (min, max) blob sizes
    - threshold: threshold for creating gaps (lower = more gaps)
    """
    ny, nx = shape
    field = np.zeros((ny, nx))
    
    # Get valid pixel locations
    valid_locs = np.where(tempo_mask)
    if len(valid_locs[0]) == 0:
        # No valid pixels, return empty mask
        return np.zeros(shape, dtype=np.float32)
    
    # Large-scale cloud structures - more elongated
    for _ in range(n_blobs):
        # Random blob center from valid locations only
        idx = np.random.randint(0, len(valid_locs[0]))
        cy = valid_locs[0][idx]
        cx = valid_locs[1][idx]
        
        # Random blob size (much larger)
        blob_size = np.random.randint(*blob_size_range)
        
        # Create elongated cloud-like blob
        y, x = np.ogrid[:ny, :nx]
        
        # Make more elongated by reducing aspect ratio range
        angle = np.random.uniform(0, 2*np.pi)
        aspect_ratio = np.random.uniform(0.1, 0.4)  # More elongated (was 0.3, 1.0)
        
        # Rotate coordinates
        y_rot = (y - cy) * np.cos(angle) - (x - cx) * np.sin(angle)
        x_rot = (y - cy) * np.sin(angle) + (x - cx) * np.cos(angle)
        
        # Create highly elliptical blob
        blob = np.exp(-(y_rot**2 / (2 * (blob_size/1.5)**2) + 
                       (x_rot * aspect_ratio)**2 / (2 * (blob_size/4)**2)))
        
        # Add to field with random intensity
        field += blob * np.random.uniform(0.8, 1.5)
    
    # Add medium-scale details - also elongated
    for _ in range(n_blobs * 2):
        # Random center from valid locations
        idx = np.random.randint(0, len(valid_locs[0]))
        cy = valid_locs[0][idx]
        cx = valid_locs[1][idx]
        
        blob_size = np.random.randint(40, 100)
        
        y, x = np.ogrid[:ny, :nx]
        
        # Make medium blobs elongated too
        angle = np.random.uniform(0, 2*np.pi)
        aspect_ratio = np.random.uniform(0.02, 0.06)
        
        y_rot = (y - cy) * np.cos(angle) - (x - cx) * np.sin(angle)
        x_rot = (y - cy) * np.sin(angle) + (x - cx) * np.cos(angle)
        
        blob = np.exp(-(y_rot**2 / (2 * (blob_size/3)**2) + 
                       (x_rot * aspect_ratio)**2 / (2 * (blob_size/5)**2)))
        
        field += blob * np.random.uniform(0.3, 0.8)
    
    # Add fine-scale texture
    for _ in range(n_blobs * 4):
        # Random center from valid locations
        idx = np.random.randint(0, len(valid_locs[0]))
        cy = valid_locs[0][idx]
        cx = valid_locs[1][idx]
        
        blob_size = np.random.randint(15, 40)
        
        y, x = np.ogrid[:ny, :nx]
        blob = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * (blob_size/5)**2))
        field += blob * np.random.uniform(0.1, 0.5)
    
    # Apply stronger Gaussian smoothing for more diffuse appearance
    field = gaussian_filter(field, sigma=5)
    
    # Add some noise for more realistic texture
    noise = np.random.normal(0, 0.1, field.shape)
    field += gaussian_filter(noise, sigma=3)
    
    # Normalize and apply threshold
    field = (field - field.min()) / (field.max() - field.min())
    gap_mask = (field < threshold).astype(np.float32)
    
    # Constrain gaps to only occur where TEMPO pixels are valid
    gap_mask = gap_mask * tempo_mask.astype(np.float32)
    
    return gap_mask