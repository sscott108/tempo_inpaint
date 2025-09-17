import os
import re
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
import rasterio
import pandas as pd
import torch
import matplotlib.pyplot as plt

from ..utils.helpers import extract_timestamp_from_path, load_shapefile_segments_pyshp, generate_realistic_gaps_simple


class TempoInpaintDataset(Dataset):
    F32_MIN = np.float32(-3.4028235e+38).item()
    def __init__(self,
                 tif_dir,
                 normalizer,
                 file_list,
                 train,
                 use_dataset_mask=True,
                 treat_zeros_as_missing=False,
                 valid_range=None):
        self.tif_dir = tif_dir
        self.train= train
        self.normalizer = normalizer
        self.use_dataset_mask = bool(use_dataset_mask)
        self.treat_zeros_as_missing = bool(treat_zeros_as_missing)
        self.valid_range = valid_range

        # store and index files by timestamp
        self.files = list(file_list)
        self.timestamps = []
        for p in self.files:
            ts = self._parse_time_from_fname(os.path.basename(p))
            self.timestamps.append(ts)

        # sort by time (keep a parallel array of paths)
        order = np.argsort(np.array(self.timestamps, dtype='datetime64[ns]'))
        self.files = [self.files[i] for i in order]
        self.timestamps = [self.timestamps[i] for i in order]

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
            
    def sample_vis(self, idx=None, train=None):
        if idx is None:
            idx = np.random.randint(len(self))

        # Use the dataset's train flag if not explicitly provided
        if train is None:
            train = self.train

        sample = self[idx]

        inp_np = sample["masked_img"][0].numpy()
        known_mask = sample["known_mask"][0].numpy().astype(bool)
        
        targ = sample['target'][0].numpy()

        # Calculate vmin/vmax for consistent scaling
        vmin, vmax = np.percentile(targ[np.isfinite(targ)], [2, 98])
        cmap_v = plt.cm.viridis.copy()
        cmap_v.set_bad("white")

        if train:
            # For training dataset, we have the realistic gaps
            realistic_gaps = sample['fake_mask'][0].numpy().astype(bool)
            all_masks = sample["known_and_fake_mask"][0].numpy().astype(bool)
            # SEPARATE THE GAPS: Extract only the gap pixels (not the original missing data)
            gaps_only_mask = known_mask & (~realistic_gaps)  # Areas that are known but made into gaps

            # Print statistics
            original_missing = (~known_mask).sum()
            gaps_only = gaps_only_mask.sum()
            all_artificial = (~realistic_gaps).sum()
            total_holes = (~all_masks).sum()

            gap_percentage= 100 * gaps_only / known_mask.sum()
                
            
            fig, ax = plt.subplots(1, 5, figsize=(26, 6))

            # Panel 1: Input (masked with all masks)
            im0 = ax[0].imshow(np.ma.array(inp_np, mask=~all_masks), 
                              cmap=cmap_v, vmin=vmin, vmax=vmax)
            ax[0].set_title("Input (all masks)")
            fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            # Panel 2: Original missing data (sensor holes)
            ax[1].imshow(known_mask == 0, cmap="Reds", alpha=0.7)
            ax[1].set_title("Original Missing Data")

            # Panel 3: GAPS ONLY (artificial holes in previously known areas)
            ax[2].imshow(gaps_only_mask, cmap="Blues", alpha=0.7)
            ax[2].set_title(f"Gaps Only {gap_percentage:.1f}")


            # Panel 4: Combined mask (all holes)
            ax[3].imshow(all_masks == 0, cmap="gray", alpha=0.7)
            ax[3].set_title("All Masks Combined")

            # Panel 6: Original target
            im4 = ax[4].imshow(np.ma.array(targ, mask=~np.isfinite(targ)),
                              cmap=cmap_v, vmin=vmin, vmax=vmax)
            ax[4].set_title("Original Target")
            fig.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

            for a in ax: 
                a.axis("off")
            plt.tight_layout()
            plt.show()

            print(f"Original missing data: {original_missing} pixels")
            print(f"Gaps only (artificial in known areas): {gaps_only} pixels")
            print(f"All artificial holes: {all_artificial} pixels")
            print(f"Total holes: {total_holes} pixels")
            print(f"Gap percentage of known data: {gap_percentage:.1f}%")

        else:
            # Validation dataset: show only 3 panels (no artificial gaps)
            fig, ax = plt.subplots(1, 3, figsize=(15, 6))

            # Panel 1: Input (masked with known mask only)
            im0 = ax[0].imshow(np.ma.array(inp_np, mask=~known_mask), 
                              cmap=cmap_v, vmin=vmin, vmax=vmax)
            ax[0].set_title("Input (original mask)")
            fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            # Panel 2: Original missing data
            ax[1].imshow(known_mask == 0, cmap="Reds", alpha=0.7)
            ax[1].set_title("Original Missing Data")

            # Panel 3: Original target
            im2 = ax[2].imshow(np.ma.array(targ, mask=~np.isfinite(targ)),
                              cmap=cmap_v, vmin=vmin, vmax=vmax)
            ax[2].set_title("Original Target")
            fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            for a in ax: 
                a.axis("off")
            plt.tight_layout()
            plt.show()

            # Print statistics
            print(f"Original missing data: {(~known_mask).sum()} pixels")
            print(f"Total pixels: {known_mask.size}")
            print(f"Missing percentage: {100 * (~known_mask).sum() / known_mask.size:.1f}%")
        
    # ---------- torch Dataset API ----------
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):     
        path = self.files[idx]
        arr_valid, known_mask = self._read_band_masked(path)
        img = np.nan_to_num(arr_valid, nan=0.0).astype(np.float64)
        H, W = img.shape
        img_n = self.normalizer.normalize_image(img) 
    
    
    
        if self.train:
            n_blobs= np.random.randint(0,5)
            realistic_gaps = generate_realistic_gaps_simple(
                                    shape=(H, W), 
                                    tempo_mask=(known_mask),  # Your TEMPO valid pixel mask
                                    n_blobs=n_blobs, 
                                    blob_size_range=(20, 74),  # Can make even larger
                                    threshold=0.6
                                )

            all_masks = known_mask * realistic_gaps
            img_with_holes = img_n * all_masks

            sample = {
                "masked_img": torch.from_numpy(img_with_holes).unsqueeze(0).float(),         #input image to model with all holes real and / or fake
                "known_and_fake_mask": torch.from_numpy(all_masks).unsqueeze(0).float(),       # mask used in training, 
                "known_mask": torch.from_numpy(known_mask).unsqueeze(0).float(),            # real missing pixels only, 1=pixel available, 0=no pixel available
                "fake_mask": torch.from_numpy(realistic_gaps).unsqueeze(0).float(),          # salt/pepper holes
                "target": torch.from_numpy(img_n).unsqueeze(0).float(),                  #image normed alone
                "path": path,
            }
            return sample

        else:
            sample = {
                "masked_img": torch.from_numpy(img_n).unsqueeze(0).float(),         #input image to model with all holes real and / or fake
                "known_mask": torch.from_numpy(known_mask).unsqueeze(0).float(),            # real missing pixels only, 1=pixel available, 0=no pixel available
                "target": torch.from_numpy(img_n).unsqueeze(0).float(),                  #image normed alone
                "path": path}
        
            return sample
        
        
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
                time_tolerance="30min"):
        self.tif_dir = tif_dir
        self.train= train
        self.normalizer = normalizer
        self.use_dataset_mask = bool(use_dataset_mask)
        self.treat_zeros_as_missing = bool(treat_zeros_as_missing)
        self.valid_range = valid_range

        # store and index files by timestamp
        self.files = list(file_list)
        self.timestamps = []
        for p in self.files:
            ts = self._parse_time_from_fname(os.path.basename(p))
            self.timestamps.append(ts)

        # sort by time (keep a parallel array of paths)
        order = np.argsort(np.array(self.timestamps, dtype='datetime64[ns]'))
        self.files = [self.files[i] for i in order]
        self.timestamps = [self.timestamps[i] for i in order]
        
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
            
    def sample_vis(self, idx=None, train=None, shp_path='/work/srs108/pconv2d/cus/cb_2018_us_state_500k.shp"', seed=42):
        if idx is None:
            idx = np.random.randint(len(self))

        # Use the dataset's train flag if not explicitly provided
        if train is None:
            train = self.train

        sample = self[idx]

        inp_np = sample["masked_img"][0].numpy()
        known_mask = sample["known_mask"][0].numpy().astype(bool)
        targ = sample['target'][0].numpy()

        # Get Pandora data if available
        p_mask = sample.get('p_mask', None)
        p_val_map = sample.get('p_val_map', None)

        # Calculate vmin/vmax for consistent scaling
        vmin, vmax = np.percentile(targ[np.isfinite(targ)], [2, 98])
        cmap_v = plt.cm.viridis.copy()
        cmap_v.set_bad("white")

        # Get image metadata for Pandora plotting
        path = sample["path"]
        date_str = path.split('/')[-1].split('.')[0]
        date = datetime.strptime(date_str, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

        # Get georeferencing info for Pandora
        with rasterio.open(path) as src:
            tr = src.transform
            crs = src.crs
            H, W = src.height, src.width

        ts = self._parse_time_from_fname(os.path.basename(path))
        xmin, ymin, xmax, ymax = array_bounds(H, W, tr)

        # Load shapefile segments if provided
        segments = []
        if shp_path:
            segments = load_shapefile_segments_pyshp(shp_path, crs)

        def _add_shape_and_pandora(ax, plot_pandora=True):
            """Helper function to add shapefile and Pandora stations"""
            # Add shapefile
            if segments:
                ax.add_collection(LineCollection(segments, colors='k', linewidths=0.5, zorder=3))

            # Add Pandora stations if available and requested
            if plot_pandora and hasattr(self, 'pandora_df') and self.pandora_df is not None and ts is not pd.NaT:
                # Filter Pandora data for this timestamp
                dfw = self.pandora_df[
                    (self.pandora_df["datetime"] >= ts - self.time_tolerance) &
                    (self.pandora_df["datetime"] <= ts + self.time_tolerance)
                ].copy()

                if not dfw.empty:
                    if "station" in dfw.columns:
                        dfw["abs_dt"] = (dfw["datetime"] - ts).abs()
                        dfw = dfw.sort_values(["station", "abs_dt"]).groupby("station", as_index=False).first()

                    # Create color map for stations
                    stations_all = self.pandora_df["station"].unique()
                    color_map = dict(zip(stations_all, cm.tab20c(np.linspace(0, 1, len(stations_all)))))

                    # Get coordinates
                    lons = _wrap_lon_180(pd.to_numeric(dfw["lon"], errors="coerce").to_numpy())
                    lats = pd.to_numeric(dfw["lat"], errors="coerce").to_numpy()
                    ok_ll = np.isfinite(lons) & np.isfinite(lats) & (lats >= -90) & (lats <= 90)

                    if ok_ll.sum() > 0:
                        dfw = dfw.loc[ok_ll].copy()
                        rr, cc = _lonlat_to_rowcol_vec(lons[ok_ll], lats[ok_ll], tr, crs)
                        labels = dfw["station"].astype(str).to_numpy()

                        rr_i = rr.astype(int)
                        cc_i = cc.astype(int)
                        ok_in = (rr_i >= 0) & (rr_i < H) & (cc_i >= 0) & (cc_i < W)
                        rr_i, cc_i, labels = rr_i[ok_in], cc_i[ok_in], labels[ok_in]

                        if rr_i.size > 0:
                            xs, ys = _rowcol_to_xy_vec(rr_i, cc_i, tr)

                            for x, y, lab in zip(xs, ys, labels):
                                c = color_map.get(lab, "red")

                                # Plot with white halo + colored marker + black edge
                                ax.scatter(x, y, s=110, marker='D', color='white', zorder=4, linewidths=0)
                                sc = ax.scatter(x, y, s=80, marker='D', color=c, edgecolor='k', linewidth=0.8, zorder=5)
                                sc.set_path_effects([pe.Stroke(linewidth=1.4, foreground='white'), pe.Normal()])

                            return True, color_map, labels  # Return info for legend

            return False, {}, []

        if train:
            # For training dataset, we have the realistic gaps
            realistic_gaps = sample['fake_mask'][0].numpy().astype(bool)
            all_masks = sample["known_and_fake_mask"][0].numpy().astype(bool)

            # SEPARATE THE GAPS: Extract only the gap pixels (not the original missing data)
            gaps_only_mask = known_mask & (~realistic_gaps)  # Areas that are known but made into gaps

            # Print statistics
            original_missing = (~known_mask).sum()
            gaps_only = gaps_only_mask.sum()
            all_artificial = (~realistic_gaps).sum()
            total_holes = (~all_masks).sum()
            gap_percentage = 100 * gaps_only / known_mask.sum()

            fig, ax = plt.subplots(1, 4, figsize=(26, 6))

            # Panel 1: Input (masked with all masks) + Pandora
            im0 = ax[0].imshow(np.ma.array(inp_np, mask=~all_masks), 
                              cmap=cmap_v, vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
            ax[0].set_title(f"Input (all masks)\n{date}")
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(ymin, ymax)
            ax[0].set_aspect('equal')
            fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            # Add shapefile and Pandora to panel 1
            has_pandora, color_map, station_labels = _add_shape_and_pandora(ax[0], plot_pandora=True)

#             # Panel 2: Original missing data (sensor holes)
#             ax[1].imshow(known_mask == 0, cmap="Reds", alpha=0.7)
#             ax[1].set_title("Original Missing Data")

            # Panel 3: GAPS ONLY (artificial holes in previously known areas)
            ax[1].imshow(gaps_only_mask, cmap="Blues", alpha=0.7)
            ax[1].set_title(f"Gaps Only {gap_percentage:.1f}%")

            # Panel 4: Combined mask (all holes)
            ax[2].imshow(all_masks == 0, cmap="gray", alpha=0.7)
            ax[2].set_title("All Masks Combined")

            # Panel 5: Original target + Pandora
            im4 = ax[3].imshow(np.ma.array(targ, mask=~np.isfinite(targ)),
                              cmap=cmap_v, vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
            ax[3].set_title(f"Original Target\n{date}")
            ax[3].set_xlim(xmin, xmax)
            ax[3].set_ylim(ymin, ymax)
            ax[3].set_aspect('equal')
            fig.colorbar(im4, ax=ax[3], fraction=0.046, pad=0.04)

            # Add shapefile and Pandora to panel 5
            _add_shape_and_pandora(ax[3], plot_pandora=True)

            # Add legend if Pandora stations were plotted
            if has_pandora and len(station_labels) > 0:
                legend_handles = []
                for lab in np.unique(station_labels):
                    c = color_map.get(lab, "red")
                    proxy = Line2D([0], [0], marker='D', color='none',
                                 markerfacecolor=c, markeredgecolor='k', markeredgewidth=0.8,
                                 markersize=9, label=lab)
                    legend_handles.append(proxy)

                ax[0].legend(handles=legend_handles, bbox_to_anchor=(-0.85, 1),
                            loc="upper left", borderaxespad=0., frameon=True,
                            fontsize=10, markerscale=1.2)

            for a in ax[1:4]:  # Skip panels with geographic coords
                a.axis("off")

            plt.tight_layout()
            plt.show()

        else:
            # Validation dataset: show only 3 panels (no artificial gaps)
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))

            # Panel 1: Input (masked with known mask only) + Pandora
            im0 = ax[0].imshow(np.ma.array(inp_np, mask=~known_mask), 
                              cmap=cmap_v, vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
            ax[0].set_title(f"Input (original mask)\n{date}")
            ax[0].set_xlim(xmin, xmax)
            ax[0].set_ylim(ymin, ymax)
            ax[0].set_aspect('equal')
            fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

            # Add shapefile and Pandora to panel 1
            has_pandora, color_map, station_labels = _add_shape_and_pandora(ax[0], plot_pandora=True)

            # Panel 2: Original missing data
            ax[1].imshow(known_mask == 0, cmap="Reds", alpha=0.7)
            ax[1].set_title("Original Missing Data")

            # Panel 3: Original target + Pandora
            im2 = ax[2].imshow(np.ma.array(targ, mask=~np.isfinite(targ)),
                              cmap=cmap_v, vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
            ax[2].set_title(f"Original Target\n{date}")
            ax[2].set_xlim(xmin, xmax)
            ax[2].set_ylim(ymin, ymax)
            ax[2].set_aspect('equal')
            fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

            # Add shapefile and Pandora to panel 3
            _add_shape_and_pandora(ax[2], plot_pandora=True)

            # Add legend if Pandora stations were plotted
            if has_pandora and len(station_labels) > 0:
                legend_handles = []
                for lab in np.unique(station_labels):
                    c = color_map.get(lab, "red")
                    proxy = Line2D([0], [0], marker='D', color='none',
                                 markerfacecolor=c, markeredgecolor='k', markeredgewidth=0.8,
                                 markersize=9, label=lab)
                    legend_handles.append(proxy)

                ax[0].legend(handles=legend_handles, bbox_to_anchor=(-0.85, 1),
                            loc="upper left", borderaxespad=0., frameon=True,
                            fontsize=10, markerscale=1.2)

            ax[1].axis("off")  # Only turn off axis for the middle panel

            plt.tight_layout()
            plt.show()
        
    # ---------- torch Dataset API ----------
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

                # optional normalization for Pandora values
                if hasattr(self.normalizer, "normalize_pandora_array"): vals_n = self.normalizer.normalize_pandora_array(vals.astype(np.float64)).astype(np.float32)
                elif hasattr(self.normalizer, "normalize_pandora"): vals_n = np.array([self.normalizer.normalize_pandora(v) for v in vals], dtype=np.float32)
                else: vals_n = vals.astype(np.float32)  # identity

                for r, c, v_n in zip(rows, cols, vals_n):
                    pandora_mask[r, c] = 1.0
                    pandora_val_map[r, c] = v_n
                    xy_list.append((int(r), int(c)))
                val_list = vals_n.tolist()
                
        if self.train:
            n_blobs= np.random.randint(0,5)
            realistic_gaps = generate_realistic_gaps_simple(
                                    shape=(H, W), 
                                    tempo_mask=(known_mask),  # Your TEMPO valid pixel mask
                                    n_blobs=n_blobs, 
                                    blob_size_range=(20, 74),  # Can make even larger
                                    threshold=0.6
                                )

            all_masks = known_mask * realistic_gaps
            img_with_holes = img_n * all_masks
                        

            sample = {
                "p_mask": torch.from_numpy(pandora_mask),
                "p_val_mask": torch.from_numpy(pandora_val_map),
                "masked_img": torch.from_numpy(img_with_holes).unsqueeze(0).float(),         #input image to model with all holes real and / or fake
                "known_and_fake_mask": torch.from_numpy(all_masks).unsqueeze(0).float(),       # mask used in training, 
                "known_mask": torch.from_numpy(known_mask).unsqueeze(0).float(),            # real missing pixels only, 1=pixel available, 0=no pixel available
                "fake_mask": torch.from_numpy(realistic_gaps).unsqueeze(0).float(),          # salt/pepper holes
                "target": torch.from_numpy(img_n).unsqueeze(0).float(),                  #image normed alone
                "path": path,
            }
            return sample

        else:
            sample = {
                "p_mask": torch.from_numpy(pandora_mask),
                "p_val_mask": torch.from_numpy(pandora_val_map),
                "masked_img": torch.from_numpy(img_n).unsqueeze(0).float(),         #input image to model with all holes real and / or fake
                "known_mask": torch.from_numpy(known_mask).unsqueeze(0).float(),            # real missing pixels only, 1=pixel available, 0=no pixel available
                "target": torch.from_numpy(img_n).unsqueeze(0).float(),                  #image normed alone
                "path": path}
        
            return sample
        
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