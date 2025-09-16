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

def visualize_batch(epoch, model, dataset, idx=19, device="cuda", shp_path=None, avg_thr=0.5):
    """
    Visualize model predictions with optional shapefile overlay
    """
    model.eval()
    sample = dataset[idx]
    
    # Get prediction
    with torch.no_grad():
        img = sample["img_w_both_masks"].unsqueeze(0).to(device)
        mask = sample["known_and_fake_mask"].unsqueeze(0).to(device)
        pred, pred_mask = model(img, mask)
        pred_np = pred.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy().astype(bool)
    
    # Extract metadata if shapefile provided
    if shp_path:
        path = sample["path"]
        date = extract_timestamp_from_path(path)
        
        with rasterio.open(path) as src:
            tr = src.transform
            crs = src.crs
            H, W = src.height, src.width
            xmin, ymin, xmax, ymax = src.bounds

        segments = _load_shapefile_segments_pyshp(shp_path, crs)
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

    # Calculate gradients for texture analysis
    try:
        gy = ndi.sobel(pred_np, axis=0)
        gx = ndi.sobel(pred_np, axis=1)
    except Exception:
        gy, gx = np.gradient(pred_np.astype(np.float32))
    grad = np.hypot(gx, gy)

    known_grad = grad[mask_np & np.isfinite(grad)]
    if known_grad.size:
        p10, p90 = np.percentile(known_grad, [10, 90])
        scale = max(p90 - p10, 1e-6)
        grad_norm = (grad - p10) / scale
    else:
        grad_norm = grad

    avg_mask = (~mask_np) & (grad_norm < float(avg_thr))  # holes with low texture
    avg_frac = 100.0 * (avg_mask.sum() / max((~mask_np).sum(), 1))

    # Set up colormap
    cmap_v = plt.cm.viridis.copy()
    cmap_v.set_bad(color="white")

    # Create visualization plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original with mask
    masked_img = sample["target"].numpy().copy()
    masked_img[~mask_np] = np.nan
    axes[0].imshow(masked_img, cmap=cmap_v)
    axes[0].set_title("Original (masked)")
    
    # Prediction
    axes[1].imshow(pred_np, cmap=cmap_v)
    axes[1].set_title("Prediction")
    
    # Difference
    diff = pred_np - sample["target"].numpy()
    axes[2].imshow(diff, cmap='RdBu_r')
    axes[2].set_title("Difference")
    
    # Add shapefile overlay if available
    if shp_path and segments:
        def _add_shape(ax, alpha=1.0, color="k"):
            ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6, zorder=6, alpha=alpha))
        
        for ax in axes:
            _add_shape(ax, alpha=0.7, color="red")
    
    plt.tight_layout()
    plt.savefig(f"visualization_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return avg_frac

def _add_shape(ax, segments, alpha=1.0, color="k"):
    """Helper function to add shapefile overlay to plot"""
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6, zorder=6, alpha=alpha))
    
def _load_shapefile_segments_pyshp(shp_path, target_crs):
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