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


def visualize_batch(epoch,model, dataset, idx, device="cuda", save=False, train = True, shp_path=None, avg_thr=0.2):
    model.eval()
    sample = dataset[idx]   # dict: {"img_in","mask_in","target","path"}
    img    = sample["img_w_both_masks"].unsqueeze(0).to(device)   # [1,1,H,W]
    mask   = sample["known_and_fake_mask"].unsqueeze(0).to(device)  # [1,1,H,W]
    mask_obs = sample["known_mask"].unsqueeze(0).to(device)  #real missing tempo pixels, no fake mask
    target = sample["target"].unsqueeze(0).to(device)   # [1,1,H,W]
    
    if train: 
        fake_mask = sample['fake_mask'].unsqueeze(0).to(device)
        fake_mask = fake_mask[0,0].cpu().numpy().astype(bool)
        
    with torch.no_grad():pred, out_mask  = model(img, mask)
    
    inp_np   = normalizer.denormalize_image(img[0,0].cpu().numpy())
    mask_np  = mask[0,0].cpu().numpy().astype(bool)   # 1=known
    pred_np  = normalizer.denormalize_image(pred[0,0].cpu().numpy())
    tgt_np   = normalizer.denormalize_image(target[0,0].cpu().numpy())
    mask_obs = mask_obs[0,0].cpu().numpy().astype(bool)
    
    holes_before = np.count_nonzero(~mask_np)
    holes_filled = np.count_nonzero((~mask_np) & np.isfinite(pred_np))
    fill_frac = 100.0 * holes_filled / max(holes_before, 1)

    path = sample["path"]
    base = os.path.basename(path)
    ts = re.sub(r"\D", "", base)[:14]
    date = datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    with rasterio.open(path) as src:
        tr = src.transform
        crs = src.crs
        H, W = src.height, src.width
        xmin, ymin, xmax, ymax = src.bounds

    segments = _load_shapefile_segments_pyshp(shp_path, crs)
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
    try:
        gy = ndi.sobel(pred_np, axis=0); gx = ndi.sobel(pred_np, axis=1)
    except Exception:
        gy, gx = np.gradient(pred_np.astype(np.float32))
    grad = np.hypot(gx, gy)

    known_grad = grad[(mask_np) & np.isfinite(grad)]
    if known_grad.size:
        p10, p90 = np.percentile(known_grad, [10, 90])
        scale = max(p90 - p10, 1e-6)
        grad_norm = (grad - p10) / scale
    else:
        grad_norm = grad

    avg_mask = (~mask_np) & (grad_norm < float(avg_thr))  # holes with low texture
    avg_frac = 100.0 * (avg_mask.sum() / max((~mask_np).sum(), 1))

    cmap_v = plt.cm.viridis.copy()
    cmap_v.set_bad(color="white")

    finite = np.isfinite(tgt_np)
    vmin, vmax = (np.percentile(tgt_np[finite], [2, 98]) if finite.any() else (0, 1))
    
    inp_ma   = np.ma.array(inp_np,   mask=~mask_np)   # holes → white
    tgt_ma   = np.ma.array(tgt_np,  mask=~np.isfinite(tgt_np))
    pred_ma  = np.ma.array(pred_np, mask=np.isfinite(pred_np))
    
    if train:
        sample_type = "Training"
        fig, ax = plt.subplots(1, 5, figsize=(15, 8))
        
        im0 = ax[0].imshow(np.ma.array(inp_np, mask=~mask_np), cmap=cmap_v, vmin=vmin, vmax=vmax,extent=(0, W, H, 0), origin="upper")
        ax[0].set_title("Input (holes = white)");_add_shape(ax[0], color="k")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(fake_mask, cmap="gray", extent=(0, W, H, 0), origin="upper")
        ax[1].set_title("Train Artificial Mask");_add_shape(ax[1], color="r")

        im2 = ax[2].imshow(np.ma.array(pred_np, mask=~np.isfinite(pred_np)),cmap=cmap_v, vmin=vmin, vmax=vmax,extent=(0, W, H, 0), origin="upper", zorder=5)
        ax[2].set_title(f"Reconstruction {avg_frac:.1f}%");_add_shape(ax[2], color="k")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        # filled_only: prediction values only in holes
        filled_only = np.full_like(pred_np, np.nan, dtype=np.float32)
        filled_only[~mask_np] = pred_np[~mask_np]   # keep values only where mask==0

        # masked array: holes show predictions, known pixels hidden
        im3 = ax[3].imshow(np.ma.array(filled_only, mask=np.isnan(filled_only)),cmap=cmap_v, vmin=vmin, vmax=vmax,extent=(0, W, H, 0), origin="upper", zorder=5)
        ax[3].set_title("Filled Values in Holes");_add_shape(ax[3], color="k");fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        im4 = ax[4].imshow(np.ma.array(tgt_np, mask=~mask_obs), cmap=cmap_v, vmin=vmin, vmax=vmax,extent=(0, W, H, 0), origin="upper")
        ax[4].set_title("Input (holes = white)");_add_shape(ax[4],color="k")
        fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)
    else:
        sample_type = "Validation"
        fig, ax = plt.subplots(1, 4, figsize = (15,7))
        
        im0 = ax[0].imshow(np.ma.array(inp_np, mask=~mask_np), cmap=cmap_v, vmin=vmin, vmax=vmax,extent=(0, W, H, 0), origin="upper")
        ax[0].set_title("Input (holes = white)");_add_shape(ax[0], color="k")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)  
        
        im1 = ax[1].imshow(mask_np, cmap="gray", extent=(0, W, H, 0), origin="upper")
        ax[1].set_title("True Missing Mask");_add_shape(ax[1], color="r")

        im2 = ax[2].imshow(np.ma.array(pred_np, mask=~np.isfinite(pred_np)),cmap=cmap_v, vmin=vmin, vmax=vmax,extent=(0, W, H, 0), origin="upper", zorder=5)
        ax[2].set_title(f"Reconstruction {avg_frac:.1f}%");_add_shape(ax[2], color="k")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        # filled_only: prediction values only in holes
        filled_only = np.full_like(pred_np, np.nan, dtype=np.float32)
        filled_only[~mask_np] = pred_np[~mask_np]   # keep values only where mask==0

        # masked array: holes show predictions, known pixels hidden
        im3 = ax[3].imshow(np.ma.array(filled_only, mask=np.isnan(filled_only)),cmap=cmap_v, vmin=vmin, vmax=vmax,extent=(0, W, H, 0), origin="upper", zorder=5)
        ax[3].set_title("Target Image");_add_shape(ax[3], color="k");fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
        
    for a in ax: a.axis("off")

    fig.suptitle(f" {sample_type} image {date}, epoch {epoch}", fontsize=16, y=0.79)
    plt.tight_layout()
    save_path = path.split('/')[-1].split('.')[0]
    if save:
        plt.savefig(f'{save_path}_{sample_type}_epoch_{epoch}.png', dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

def _add_shape(ax, segments, alpha=1.0, color="k"):
    """Helper function to add shapefile overlay to plot"""
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6, zorder=6, alpha=alpha))
    
def load_shapefile_segments_pyshp(shp_path, target_crs):
    """
    Load shapefile segments and transform to target CRS
    """
    src_crs = None  # You'll need to determine source CRS
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