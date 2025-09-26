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
from rasterio.transform import array_bounds
from rasterio.plot import show as rio_show
import matplotlib.cm as cm
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D


def _wrap_lon_180(lons):
    """Normalize longitudes to [-180, 180]."""
    arr = np.asarray(lons, dtype=float)
    m = np.isfinite(arr)
    arr[m] = ((arr[m] + 180.0) % 360.0) - 180.0
    return arr
def _lonlat_to_rowcol_vec(lons, lats, transform, raster_crs):

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    # mask invalid inputs
    good = np.isfinite(lons) & np.isfinite(lats)
    rows = np.full_like(lons, fill_value=np.nan, dtype=float)
    cols = np.full_like(lons, fill_value=np.nan, dtype=float)

    if not good.any(): return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    try:
        from pyproj import CRS, Transformer
        src = CRS.from_epsg(4326)  # Pandora files give lon/lat in WGS84
        dst = raster_crs if raster_crs is not None else CRS.from_epsg(4326)
        if not isinstance(dst, CRS):
            dst = CRS.from_user_input(dst)
        xf = Transformer.from_crs(src, dst, always_xy=True)
        xs, ys = xf.transform(lons[good], lats[good])
    except Exception:
        # Fallback: assume raster is already in lon/lat
        xs, ys = lons[good], lats[good]

    # Affine inverse: xy -> col,row
    inv = ~transform
    cc = []; rr = []
    for x, y in zip(xs, ys):
        c, r = inv * (x, y)
        cc.append(c); rr.append(r)
    cols[good] = np.round(cc)
    rows[good] = np.round(rr)

    # keep only valid converted points
    rows_i = rows[good].astype(np.int32)
    cols_i = cols[good].astype(np.int32)
    return rows_i, cols_i


def _rowcol_to_xy_vec(rows, cols, transform):
    """
    Vectorized: pixel row/col -> raster xy using the affine transform.
    """
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    xs, ys = [], []
    for r, c in zip(rows, cols):
        x, y = transform * (c, r)
        xs.append(x); ys.append(y)
    return np.asarray(xs), np.asarray(ys)

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


def _add_shape_pixel(ax, segments, tr, alpha=0.8, color="k"):
    """Add shapefile to pixel-coordinate panels"""
    if segments:
        segments_pixel = []
        for seg in segments:
            if len(seg) > 0:
                pixel_coords = []
                for x_geo, y_geo in seg:
                    col, row = ~tr * (x_geo, y_geo)
                    pixel_coords.append([col, row])
                segments_pixel.append(np.array(pixel_coords))
        ax.add_collection(LineCollection(segments_pixel, colors=color, linewidths=0.6, zorder=6, alpha=alpha))
        
def visualize_batch(epoch, model, normalizer, dataloader, batch_idx=0, sample_idx=0, device="cuda", save=False, train=True, shp_path=None, avg_thr=0.2):
    """
    Visualize model predictions from a DataLoader following the dataset's pandora setup
    """
    model.eval()
    
    # Get a batch from the dataloader
    try:
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
            sample[key] = value[sample_idx]
        elif isinstance(value, (list, tuple)):
            sample[key] = value[sample_idx]
        else:
            sample[key] = value

    # Get Pandora data from sample
    p_mask = sample.get('p_mask', torch.zeros_like(sample["known_mask"])).numpy().astype(bool)
    p_val_map = sample.get('p_val_mask', torch.zeros_like(sample["known_mask"])).numpy()
    
    # Get common data
    img = sample["masked_img"].unsqueeze(0).to(device)
    mask_obs = sample["known_mask"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    
    # For training, get additional masks
    if train:
        mask = sample["known_and_fake_mask"].unsqueeze(0).to(device)
        fake_mask = sample['fake_mask'].unsqueeze(0).to(device)
        fake_mask_np = fake_mask[0,0].cpu().numpy().astype(bool)
    else:
        mask = mask_obs
    
    # Get model prediction
    with torch.no_grad():
        pred, out_mask = model(img, mask)
    
    # Convert to numpy
    inp_np = normalizer.denormalize_image(img[0,0].cpu().numpy())
    mask_obs_np = mask_obs[0,0].cpu().numpy().astype(bool)
    pred_np = normalizer.denormalize_image(pred[0,0].cpu().numpy())
    tgt_np = normalizer.denormalize_image(target[0,0].cpu().numpy())
    
    
    from scipy.ndimage import gaussian_filter
#     pred_np_smooth = gaussian_filter(pred_np, sigma=0.8)
    
    # Only smooth the filled areas
    
    
    if train:
        mask_np = sample["known_and_fake_mask"][0].cpu().numpy().astype(bool)
    else:
        mask_np = mask_obs_np
        
    hole_mask = ~mask_np
    pred_np_final = pred_np.copy()
    pred_np_final[hole_mask] = pred_np[hole_mask]
    # Extract metadata - FOLLOWING DATASET APPROACH
    path = sample["path"]
    date = path.split('/')[-1].split('.')[0]
    date = datetime.strptime(date, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

    # Get georeferencing info - FOLLOWING DATASET APPROACH
    with rasterio.open(path) as src:
        tr = src.transform
        crs = src.crs
        H, W = src.height, src.width
    
    xmin, ymin, xmax, ymax = array_bounds(H, W, tr)

    # Load shapefile segments - FOLLOWING DATASET APPROACH
    segments = []
    if shp_path:
        try:
            segments = load_shapefile_segments_pyshp(shp_path, crs)
        except Exception as e:
            print(f"Warning: Could not load shapefile: {e}")

    # Calculate Pandora vs Prediction RMSE
    pandora_rmse = None
    pandora_rho = None
    n_pandora_stations = 0
    
    if p_mask.any():
        pandora_rows, pandora_cols = np.where(p_mask)
        if len(pandora_rows) > 0:
            pandora_values = p_val_map[pandora_rows, pandora_cols]
            
            # Denormalize Pandora values if possible
            if hasattr(normalizer, 'denormalize_pandora'):
                pandora_values_denorm = np.array([normalizer.denormalize_pandora(v) for v in pandora_values])
            elif hasattr(normalizer, 'denormalize_pandora_array'):
                pandora_values_denorm = normalizer.denormalize_pandora_array(pandora_values)
            else:
                pandora_values_denorm = pandora_values
            
            pred_at_pandora = pred_np_final[pandora_rows, pandora_cols]
            
            valid_mask = np.isfinite(pandora_values_denorm) & np.isfinite(pred_at_pandora)
            if valid_mask.sum() > 0:
                pandora_valid = pandora_values_denorm[valid_mask]
                pred_valid = pred_at_pandora[valid_mask]
                
                pandora_rmse = np.sqrt(np.mean((pandora_valid - pred_valid)**2))
                if len(pandora_valid) > 1:
                    from scipy import stats
                    pandora_rho, _ = stats.spearmanr(pandora_valid, pred_valid)
                n_pandora_stations = len(pandora_valid)

    # Calculate performance metrics
    holes_total = (~mask_np).sum()
    holes_filled_valid = ((~mask_np) & np.isfinite(pred_np_final)).sum()
    fill_percentage = 100.0 * holes_filled_valid / max(holes_total, 1)

    # Set up colormap and scaling - FOLLOWING DATASET APPROACH
    finite_vals = tgt_np[np.isfinite(tgt_np)]
    if finite_vals.size:
        vmin, vmax = np.percentile(finite_vals, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmin == vmax):
            vmin, vmax = finite_vals.min(), finite_vals.max()
    else:
        vmin, vmax = 0.0, 1.0
    
    cmap_v = plt.cm.viridis.copy()
    cmap_v.set_bad(color="white")
    
    # PANDORA STATION PLOTTING - FOLLOWING DATASET APPROACH EXACTLY
    def add_pandora_stations(ax, add_legend=False):
        """Add Pandora stations with real station names from sample"""
        legend_handles = []

        if p_mask.any():
            pandora_rows, pandora_cols = np.where(p_mask)

            if len(pandora_rows) > 0:
                # Get station names from sample
                station_names = sample.get('station_names', [f"Station_{i+1}" for i in range(len(pandora_rows))])

                # Get Pandora values (denormalized)
                pandora_values = p_val_map[pandora_rows, pandora_cols]
                if hasattr(normalizer, 'denormalize_pandora'):
                    pandora_values_denorm = np.array([normalizer.denormalize_pandora(v) for v in pandora_values])
                elif hasattr(normalizer, 'denormalize_pandora_array'):
                    pandora_values_denorm = normalizer.denormalize_pandora_array(pandora_values)
                else:
                    pandora_values_denorm = pandora_values

                # Convert row/col to x/y using transform
                xs, ys = _rowcol_to_xy_vec(pandora_rows, pandora_cols, tr)

                # Create color scheme
                colors = cm.tab20c(np.linspace(0, 1, len(pandora_rows)))

                for i, (x, y, val) in enumerate(zip(xs, ys, pandora_values_denorm)):
                    c = colors[i] if len(colors) > 1 else 'red'

                    # Plot with white halo + black edge
                    ax.scatter(x, y, s=110, marker='D', color='white', zorder=4, linewidths=0)
                    sc = ax.scatter(x, y, s=50, marker='D', color=c, edgecolor='k', linewidth=0.8, zorder=5)

                    # Add path effects for visibility
                    sc.set_path_effects([pe.Stroke(linewidth=1.4, foreground='white'), pe.Normal()])

                    if add_legend:
                        # Use real station name from sample
                        station_name = station_names[i] if i < len(station_names) else f"Station_{i+1}"

                        proxy = Line2D([0], [0], marker='D', color='none',
                                     markerfacecolor=c, markeredgecolor='k', markeredgewidth=0.8,
                                     markersize=9, label=station_name)  # Now uses real station names!
                        legend_handles.append(proxy)

        return legend_handles

    if train:
        sample_type = "tr"
        
        # Calculate gap statistics and RMSE for training
        known_mask_np = mask_obs_np
        current_mask_np = mask_np
        artificial_holes = known_mask_np & (~current_mask_np)
        gap_percentage = 100.0 * artificial_holes.sum() / max(known_mask_np.sum(), 1)
        
        # Calculate RMSE for artificial holes
        reconstruction_title = "Reconstruction"
        if artificial_holes.sum() > 0:
            pred_in_artificial = pred_np_final[artificial_holes]
            target_in_artificial = tgt_np[artificial_holes]

            valid_mask = np.isfinite(pred_in_artificial) & np.isfinite(target_in_artificial)
            if valid_mask.sum() > 0:
                pred_valid = pred_in_artificial[valid_mask]
                target_valid = target_in_artificial[valid_mask]

                rmse = np.sqrt(np.mean((pred_valid - target_valid)**2))
                from scipy import stats
                rho, _ = stats.spearmanr(pred_valid, target_valid)
                
                reconstruction_title = f"Reconstruction\nRMSE: {rmse:.4E} | ρ: {rho:.2f}"
        
        fig, ax = plt.subplots(1, 5, figsize=(25, 6))
        
        # Panel 0: Input - USING RIO_SHOW APPROACH
        disp_inp = np.ma.masked_invalid(np.ma.array(inp_np, mask=~mask_np))
        im0 = rio_show(disp_inp, transform=tr, ax=ax[0], cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[0].set_xlim(xmin, xmax); ax[0].set_ylim(ymin, ymax)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].margins(0); ax[0].autoscale(False)
        ax[0].set_title(f"Input (N/A = white)\n{date}")
        if segments:
            ax[0].add_collection(LineCollection(segments, colors='k', linewidths=0.5, zorder=3))
        legend_handles = add_pandora_stations(ax[0], add_legend=True)
        cbar0 = fig.colorbar(im0.get_images()[0], ax=ax[0], fraction=0.046, pad=0.04)
        cbar0.set_label("NO₂ (molec$\cdot$cm$^{-2}$)")

        # Panel 1: Artificial holes
        ax[1].imshow(~artificial_holes, cmap="Reds", alpha=0.8)
        ax[1].set_title(f"Artificial Holes ({gap_percentage:.1f}%)")
        _add_shape_pixel(ax[1], segments, tr)
        
        # Panel 2: Reconstruction - USING RIO_SHOW APPROACH
        disp_pred = np.ma.masked_invalid(pred_np_final)
        im2 = rio_show(disp_pred, transform=tr, ax=ax[2], cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[2].set_xlim(xmin, xmax); ax[2].set_ylim(ymin, ymax)
        ax[2].set_aspect('equal', adjustable='box')
        ax[2].margins(0); ax[2].autoscale(False)
        ax[2].set_title(reconstruction_title)
        if segments:
            ax[2].add_collection(LineCollection(segments, colors='k', linewidths=0.5, zorder=3))
        add_pandora_stations(ax[2])
        cbar2 = fig.colorbar(im2.get_images()[0], ax=ax[2], fraction=0.046, pad=0.04)
        cbar2.set_label("NO₂ (molec$\cdot$cm$^{-2}$)")

        # Panel 3: Filled values
        filled_only = np.full_like(pred_np_final, np.nan, dtype=np.float32)
        filled_only[~mask_np] = pred_np_final[~mask_np]
        ax[3].imshow(np.ma.array(filled_only, mask=np.isnan(filled_only)), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[3].set_title("Filled Values in Holes")
        fig.colorbar(ax[3].images[0], ax=ax[3], fraction=0.046, pad=0.04)
        _add_shape_pixel(ax[3], segments, tr)

        # Panel 4: Target - USING RIO_SHOW APPROACH
        disp_tgt = np.ma.masked_invalid(np.ma.array(tgt_np, mask=~mask_obs_np))
        im4 = rio_show(disp_tgt, transform=tr, ax=ax[4], cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[4].set_xlim(xmin, xmax); ax[4].set_ylim(ymin, ymax)
        ax[4].set_aspect('equal', adjustable='box')
        ax[4].margins(0); ax[4].autoscale(False)
        ax[4].set_title(f"Target (original holes = white)\n{date}")
        if segments:
            ax[4].add_collection(LineCollection(segments, colors='k', linewidths=0.5, zorder=3))
        add_pandora_stations(ax[4])
        cbar4 = fig.colorbar(im4.get_images()[0], ax=ax[4], fraction=0.046, pad=0.04)
        cbar4.set_label("NO₂ (molec$\cdot$cm$^{-2}$)")
        
        # Add legend - FOLLOWING DATASET APPROACH
        if legend_handles:
            ax[0].legend(
                handles=legend_handles,
                bbox_to_anchor=(-0.85, 1),
                loc="upper left",
                borderaxespad=0.,
                frameon=True,
                fontsize=10,
                markerscale=1.2
            )
        
        # Turn off axes for panel 1 and 3
        ax[1].axis("off")
        ax[3].axis("off")
        
    else:
        sample_type = "val"
        
        val_title = f"Reconstruction\n{date}"
        if pandora_rmse is not None:
            if pandora_rho is not None:
                val_title = f"Reconstruction\nRMSE: {pandora_rmse:.4E} | ρ: {pandora_rho:.2f}"
            else:
                val_title = f"Reconstruction\nRMSE: {pandora_rmse:.4E}"
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 6))
        
        # Panel 0: Input - USING RIO_SHOW APPROACH
        disp_inp = np.ma.masked_invalid(np.ma.array(inp_np, mask=~mask_obs_np))
        im0 = rio_show(disp_inp, transform=tr, ax=ax[0], cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[0].set_xlim(xmin, xmax); ax[0].set_ylim(ymin, ymax)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].margins(0); ax[0].autoscale(False)
        ax[0].set_title(f"Input (N/A = white)\n{date}")
        if segments:
            ax[0].add_collection(LineCollection(segments, colors='k', linewidths=0.5, zorder=3))
        legend_handles = add_pandora_stations(ax[0], add_legend=True)
        cbar0 = fig.colorbar(im0.get_images()[0], ax=ax[0], fraction=0.046, pad=0.04)
        cbar0.set_label("NO₂ (molec$\cdot$cm$^{-2}$)")
        
        # Panel 1: Reconstruction - USING RIO_SHOW APPROACH
        disp_pred = np.ma.masked_invalid(pred_np_final)
        im1 = rio_show(disp_pred, transform=tr, ax=ax[1], cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[1].set_xlim(xmin, xmax); ax[1].set_ylim(ymin, ymax)
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].margins(0); ax[1].autoscale(False)
        ax[1].set_title(val_title)
        if segments:
            ax[1].add_collection(LineCollection(segments, colors='k', linewidths=0.5, zorder=3))
        add_pandora_stations(ax[1])
        cbar1 = fig.colorbar(im1.get_images()[0], ax=ax[1], fraction=0.046, pad=0.04)
        cbar1.set_label("NO₂ (molec$\cdot$cm$^{-2}$)")
        
        # Panel 2: Filled values
        filled_only = np.full_like(pred_np_final, np.nan, dtype=np.float32)
        filled_only[~mask_obs_np] = pred_np_final[~mask_obs_np]
        ax[2].imshow(np.ma.array(filled_only, mask=np.isnan(filled_only)), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[2].set_title("Filled Values in Holes")
        fig.colorbar(ax[2].images[0], ax=ax[2], fraction=0.046, pad=0.04)
        _add_shape_pixel(ax[2], segments, tr)

        # Panel 3: Target - USING RIO_SHOW APPROACH
        disp_tgt = np.ma.masked_invalid(tgt_np)
        im3 = rio_show(disp_tgt, transform=tr, ax=ax[3], cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[3].set_xlim(xmin, xmax); ax[3].set_ylim(ymin, ymax)
        ax[3].set_aspect('equal', adjustable='box')
        ax[3].margins(0); ax[3].autoscale(False)
        ax[3].set_title(f"Target Image\n{date}")
        if segments:
            ax[3].add_collection(LineCollection(segments, colors='k', linewidths=0.5, zorder=3))
        add_pandora_stations(ax[3])
        cbar3 = fig.colorbar(im3.get_images()[0], ax=ax[3], fraction=0.046, pad=0.04)
        cbar3.set_label("NO₂ (molec$\cdot$cm$^{-2}$)")
        
        # Add legend - FOLLOWING DATASET APPROACH
        if legend_handles:
            ax[0].legend(
                handles=legend_handles,
                bbox_to_anchor=(-0.85, 1),
                loc="upper left",
                borderaxespad=0.,
                frameon=True,
                fontsize=10,
                markerscale=1.2
            )
        
        # Turn off axes for panel 2
        ax[2].axis("off")

    # Enhanced suptitle
    title = f"{sample_type} image, epoch {epoch}"
    if n_pandora_stations > 0:
        title += f" | {n_pandora_stations} Pandora stations"
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    save_path = path.split('/')[-1].split('.')[0]
    if save:
        plt.savefig(f'{save_path}_{sample_type}_epoch_{epoch}.png', dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    
    # Print metrics
#     if pandora_rmse is not None:
#         print(f"Pandora RMSE: {pandora_rmse:.4E} ({n_pandora_stations} stations)")
#         if pandora_rho is not None:
#             print(f"Pandora correlation (ρ): {pandora_rho:.3f}")
    
    return pandora_rho, pandora_rmse