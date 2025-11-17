import os
import re
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
import rasterio
import zlib
import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy
from ..utils.helpers import extract_timestamp_from_path, load_shapefile_segments_pyshp, generate_realistic_gaps_simple



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


class TempoPandoraInpaintDataset(Dataset):
    F32_MIN = np.float32(-3.4028235e+38).item()
    def __init__(self,
                 tif_dir,
                 normalizer,
                 file_list,
                 train,
                 use_dataset_mask=True,
                 treat_zeros_as_missing=False,
                 valid_range=None,
                pandora_csv=None,
                time_tolerance="30min",
                n_blobs_range=(1, 5),
                sigma_xy_range=(5.0, 25.0),  # Spatial smoothing range
                thr_range=(0.3, 0.7),       # Threshold range for hole creation
                use_blob_gaps=True):
        
        self.tif_dir = tif_dir
        self.train= train
        self.normalizer = normalizer
        self.use_dataset_mask = bool(use_dataset_mask)
        self.treat_zeros_as_missing = bool(treat_zeros_as_missing)
        self.valid_range = valid_range
        self.files = list(file_list)
        self.timestamps = []
        for p in self.files:
            ts = self._parse_time_from_fname(os.path.basename(p))
            self.timestamps.append(ts)
            
        order = np.argsort(np.array(self.timestamps, dtype='datetime64[ns]'))
        self.files = [self.files[i] for i in order]
        self.timestamps = [self.timestamps[i] for i in order]
        
        self.n_blobs_range = n_blobs_range
        self.sigma_xy_range = sigma_xy_range
        self.thr_range = thr_range
        self.use_blob_gaps = use_blob_gaps
        self.current_epoch = 0
        self.max_epochs = 1  # will be updated in training loop

        
        # --------- Pandora table ----------
        self.pandora_df = None
        self.time_tolerance = pd.Timedelta(time_tolerance)
        if pandora_csv is not None:
            if isinstance(pandora_csv, pd.DataFrame):
                df = pandora_csv.copy()
            else:
                df = pd.read_csv(pandora_csv)

            req = {"datetime","NO2"}
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            self.pandora_df = df
            
            
    # ---------- Telea helper ----------
    def _compute_telea(self, img_normed, mask):
        """
        Compute Telea inpainting on a normalized image (0–1 or arbitrary range)
        and binary mask (1=known, 0=hole).
        Returns float32 tensor scaled back to same dynamic range.
        """
        # Normalize to 0–255 for OpenCV
        img_8u = np.clip(255 * (img_normed - img_normed.min()) / (img_normed.ptp() + 1e-8), 0, 255).astype(np.uint8)
        mask_8u = ((1 - mask) * 255).astype(np.uint8)  # Telea expects 255 for holes

        # Telea inpainting (radius = 3 is usually robust)
        inpaint_telea = cv2.inpaint(img_8u, mask_8u, 3, cv2.INPAINT_TELEA)

        # Back to float32, rescale to 0–1
        telea_pred = inpaint_telea.astype(np.float32) / 255.0
        return telea_pred

        
    # ---------- core I/O & masking ----------
    def _read_band_masked(self, path):
        with rasterio.open(path) as src:
            gdal_known = None
            if self.use_dataset_mask:
                try:
                    m = src.read_masks(1)  # 0=missing, 255=valid
                    if m is not None and m.size:
                        gdal_known = (m > 0)
                except Exception:
                    gdal_known = None

            arr = src.read(1, out_dtype='float64')
            nodatas = []
            if getattr(src, "nodata", None) is not None:
                nodatas.append(src.nodata)
            if getattr(src, "nodatavals", None):
                nodatas += [v for v in src.nodatavals if v is not None]

        known = gdal_known if gdal_known is not None else np.isfinite(arr)
        for nv in nodatas:
            known &= arr != nv
        known &= arr > (self.F32_MIN * 0.9)
        if self.treat_zeros_as_missing:
            known &= arr != 0.0
        if self.valid_range is not None:
            vmin, vmax = self.valid_range
            known &= (arr >= vmin) & (arr <= vmax)

        arr_valid = np.where(known, arr, np.nan)
        return arr_valid, known.astype(np.uint8)

    # ---------- time parsing ----------
    @staticmethod
    def _parse_time_from_fname(fname):
        digits = re.sub(r"\D", "", fname)
        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H", "%Y%m%d"):
            try:
                return pd.to_datetime(digits[:len(pd.Timestamp.now().strftime(fmt))], format=fmt)
            except Exception:
                continue
        return pd.NaT
            
    
    def _make_2d_blob_mask(
        self, H, W, keep_mask_2d, rng,
        min_frac=0.05, max_frac=0.25,              # desired coverage bounds (0–1)
        n_blobs_range=(1, 4),                      # how many “clouds” to sum
        sigma_xy_range=(8, 30),                    # smoothing (pixels) for 400×400
        multiscale=True                            # add a second, coarser octave
    ):

        n_blobs = int(rng.integers(n_blobs_range[0], n_blobs_range[1]+1))
        if n_blobs <= 0:
            return np.ones((H, W), dtype=np.float32)

        field = np.zeros((H, W), dtype=np.float32)
        for _ in range(n_blobs):
            noise = rng.standard_normal((H, W)).astype(np.float32)
            sigma = float(rng.uniform(*sigma_xy_range))
            field += scipy.ndimage.gaussian_filter(noise, sigma=sigma, mode="nearest")

            if multiscale:
                # a coarser octave makes blobs more cloud-like
                noise2 = rng.standard_normal((H, W)).astype(np.float32)
                sigma2 = sigma * 1.8
                field += 0.6 * scipy.ndimage.gaussian_filter(noise2, sigma=sigma2, mode="nearest")

        fmin, fmax = np.min(field), np.max(field)
        if fmax > fmin:
            field = (field - fmin) / (fmax - fmin)
        else:
            field = np.full((H, W), 0.5, dtype=np.float32)

        # ----- 2) Choose a target fraction and compute quantile threshold -----
        maskable = (keep_mask_2d.astype(bool))
        vals = field[maskable]
        if vals.size == 0:
            return np.ones((H, W), dtype=np.float32)

        target_frac = float(rng.uniform(min_frac, max_frac))  # variability per image
        # threshold so that ~target_frac of maskable pixels become holes
        thr = np.quantile(vals, 1.0 - target_frac)

        # Create holes & cap to max_frac (robust to numerical drift)
        holes = (field >= thr) & maskable
        # If we overshoot slightly, tighten threshold to max_frac
        if holes.mean() > max_frac:
            thr_cap = np.quantile(vals, 1.0 - max_frac)
            holes = (field >= thr_cap) & maskable

        # Convert to keep mask
        keep = np.ones((H, W), dtype=np.float32)
        keep[holes] = 0.0

        # Optional: soften edges a bit (looks more like clouds)
        # keep = scipy.ndimage.gaussian_filter(keep, sigma=0.8)
        # keep = (keep > 0.5).astype(np.float32)

        return keep
    def _make_progressive_mask(self, H, W, keep_mask_2d, rng, epoch, max_epochs):
        # compute progression factor
        frac = min(epoch / max_epochs, 1.0)  # goes 0 → 1
        # interpolate mask coverage
        max_cover = 0.25  # 25% max
        min_cover = 0.05  # 5% start
        cover_frac = min_cover + frac * (max_cover - min_cover)

        # call your _make_2d_blob_mask with threshold tuned to hit cover_frac
        keep = self._make_2d_blob_mask(H, W, keep_mask_2d, rng)
        # optionally re-balance threshold until keep.mean() ≈ 1 - cover_frac
        return keep
    
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):     
        path = self.files[idx]
        arr_valid, known_mask = self._read_band_masked(path)
        img = np.nan_to_num(arr_valid, nan=0.0).astype(np.float64)
        H, W = img.shape
        img_n = self.normalizer.normalize_image(img) 


        # ---------- Pandora anchors ----------
        pandora_mask = np.zeros((H, W), dtype=np.float32)
        pandora_val_map = np.zeros((H, W), dtype=np.float32)
        xy_list, val_list = [], []
        station_names = []

        ts = self._parse_time_from_fname(os.path.basename(path))
        if (self.pandora_df is not None) and (ts is not pd.NaT):
            dfw = self.pandora_df[
                (self.pandora_df["datetime"] >= ts - self.time_tolerance) &
                (self.pandora_df["datetime"] <= ts + self.time_tolerance)
            ].copy()

            if not dfw.empty:
                # per-station nearest in time (if station column exists)
                if "station" in dfw.columns:
                    dfw["abs_dt"] = (dfw["datetime"] - ts).abs()
                    dfw = dfw.sort_values(["station","abs_dt"]).groupby("station", as_index=False).first()

                # get row/col (prefer provided; else compute from lat/lon)
                if ("row" in dfw.columns) and ("col" in dfw.columns):
                    rows = dfw["row"].astype(int).to_numpy()
                    cols = dfw["col"].astype(int).to_numpy()
                else:
                    # compute using geotransform from this raster
                    with rasterio.open(path) as src:
                        tr = src.transform
                    if not {"lat","lon"}.issubset(set(dfw.columns)):
                        rows, cols = np.array([], dtype=int), np.array([], dtype=int)
                    else:
                        xs = dfw["lon"].to_numpy()
                        ys = dfw["lat"].to_numpy()
                        # invert: lon,lat -> row,col
                        rc = [~tr * (x, y) for x, y in zip(xs, ys)]
                        cols = np.array([int(round(c)) for c, r in rc])
                        rows = np.array([int(round(r)) for c, r in rc])

                # keep anchors inside image
                ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
                rows, cols = rows[ok], cols[ok]
                vals = dfw.loc[ok, "NO2"].astype(float).to_numpy()

                # Get station names for valid coordinates - FIXED VERSION
                dfw_ok = dfw.loc[ok].reset_index(drop=True)  # Filter dataframe to match valid coordinates
                station_names_raw = dfw_ok["station"].astype(str).to_numpy() if "station" in dfw_ok.columns else []

                # optional normalization for Pandora values
                if hasattr(self.normalizer, "normalize_pandora_array"): 
                    vals_n = self.normalizer.normalize_pandora_array(vals.astype(np.float64)).astype(np.float32)
                elif hasattr(self.normalizer, "normalize_pandora"): 
                    vals_n = np.array([self.normalizer.normalize_pandora(v) for v in vals], dtype=np.float32)
                else: 
                    vals_n = vals.astype(np.float32)  # identity

                # Store Pandora data and station names
                for i, (r, c, v_n) in enumerate(zip(rows, cols, vals_n)):
                    pandora_mask[r, c] = 1.0
                    pandora_val_map[r, c] = v_n
                    xy_list.append((int(r), int(c)))

                    # Add corresponding station name
                    if i < len(station_names_raw):
                        station_names.append(station_names_raw[i])
                    else:
                        station_names.append(f"Unknown_{i}")

                val_list = vals_n.tolist()

 # ---------- TRAIN ----------
        if self.train:
            if self.use_blob_gaps:
                rng = np.random.default_rng()
                realistic_gaps = self._make_progressive_mask(
                    H, W, known_mask, rng,
                    epoch=self.current_epoch, max_epochs=self.max_epochs
                )
            else:
                realistic_gaps = np.ones_like(known_mask, dtype=np.float32)

            all_masks = known_mask * realistic_gaps
            img_with_holes = img_n * all_masks

            # --- Telea inpainting using augmented mask ---
            telea_pred = self._compute_telea(img_n, all_masks)
            telea_pred_t = torch.from_numpy(telea_pred).unsqueeze(0).float()

            sample = {
                "p_mask": torch.from_numpy(pandora_mask),
                "p_val_mask": torch.from_numpy(pandora_val_map),
                "station_names": station_names,
                "masked_img": torch.from_numpy(img_with_holes).unsqueeze(0).float(),
                "known_and_fake_mask": torch.from_numpy(all_masks).unsqueeze(0).float(),
                "known_mask": torch.from_numpy(known_mask).unsqueeze(0).float(),
                "fake_mask": torch.from_numpy(realistic_gaps).unsqueeze(0).float(),
                "target": torch.from_numpy(img_n).unsqueeze(0).float(),
                "telea_pred": telea_pred_t,
                "path": path,
            }
            return sample

        else:
            # No augmentation → use the real known mask only
            all_masks = known_mask.astype(np.float32)
            img_with_holes = img_n * all_masks

            # --- Telea inpainting using true mask ---
            telea_pred = self._compute_telea(img_n, known_mask)
            telea_pred_t = torch.from_numpy(telea_pred).unsqueeze(0).float()

            sample = {
                "p_mask": torch.from_numpy(pandora_mask),
                "p_val_mask": torch.from_numpy(pandora_val_map),
                "station_names": station_names,
                "masked_img": torch.from_numpy(img_with_holes).unsqueeze(0).float(),
                "known_mask": torch.from_numpy(known_mask).unsqueeze(0).float(),
                "target": torch.from_numpy(img_n).unsqueeze(0).float(),
                "telea_pred": telea_pred_t,
                "path": path,
            }
            return sample