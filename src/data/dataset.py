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


    def sample_vis(self, idx=None):
        if idx is None:
            idx = np.random.randint(len(self))
        sample = self[idx]

        inp_np = sample["masked_img"][0].numpy()
        mask_obs = sample["known_mask"][0].numpy().astype(bool)
        mask_eff = sample["known_and_fake_mask"][0].numpy().astype(bool)
        targ = sample['target'][0].numpy()
        mask_sp = sample['fake_mask'][0].numpy().astype(bool)

        temp_mask = np.sum(mask_obs)
        added_mask = np.sum(mask_sp)
#         print(added_mask/temp_mask)
        vmin, vmax = np.percentile(inp_np[np.isfinite(inp_np)], [2, 98])
        cmap_v = plt.cm.viridis.copy()
        cmap_v.set_bad("white")

        fig, ax = plt.subplots(1, 5, figsize=(22, 6))

        im0 = ax[0].imshow(np.ma.array(inp_np, mask=~mask_eff), cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[0].set_title("Input (masked)");fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        ax[1].imshow(mask_obs == 0, cmap="Reds", alpha=0.7)
        ax[1].set_title("Observed Holes (sensor)")

        ax[2].imshow(mask_sp == 0, cmap="Blues", alpha=0.7)
        ax[2].set_title("Artificial Holes")
        
        ax[3].imshow(mask_eff == 0, cmap="gray", alpha=0.7)
        ax[3].set_title("Effective Mask (used in training)")
        
        im5 = ax[4].imshow(np.ma.array(targ),
                           cmap=cmap_v, vmin=vmin, vmax=vmax)
        ax[4].set_title("Regular Input")
        fig.colorbar(im5, ax=ax[4], fraction=0.046, pad=0.04)

        for a in ax: a.axis("off")
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
            "img_w_both_masks": torch.from_numpy(img_with_holes).unsqueeze(0).float(),         #input image to model with all holes real +fake
            "known_and_fake_mask": torch.from_numpy(all_masks).unsqueeze(0).float(),       # mask used in training, 
            "known_mask": torch.from_numpy(known_mask).unsqueeze(0).float(),            # real missing pixels only, 1=pixel available, 0=no pixel available
            "fake_mask": torch.from_numpy(realistic_gaps).unsqueeze(0).float(),          # salt/pepper holes
            "target": torch.from_numpy(img_n).unsqueeze(0).float(),                  #image normed alone
            "path": path,
        }
        return sample