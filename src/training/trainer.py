import torch
import torch.optim as optim
import csv
from tqdm import tqdm
import matplotlib.cm as cm
import numpy as np
from ..losses.losses import improved_loss_progress, calculate_metrics
from ..utils.helpers import visualize_sample_or_batch
import pandas as pd

pandora_df = pd.read_csv("/work/srs108/pconv2d/pandora_filtered_data_old.csv", header =0)
pandora_df["datetime"] = pd.to_datetime(pandora_df["datetime"], errors="coerce")
stations_all = pandora_df["station"].unique()
station_color_map = dict(zip(stations_all, cm.tab20c(np.linspace(0, 1, len(stations_all)))))
def train_model(model, normalizer, train_loader, val_loader, shp_path, epochs=50, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_loss = float("inf")
    best_state = None
    wait = 0
    history = []
    
    for epoch in range(epochs):
        train_loader.dataset.current_epoch = epoch
        train_loader.dataset.max_epochs = epochs
        # ---- Training ----
        model.train()
        train_loss = 0
        train_metrics = {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0,'pandora_rmse': 0.0, 'pandora_mae': 0.0, 'pandora_rho': 0.0, 'pandora_r2': 0.0,'n_pandora_stations': 0}
        train_batch_count = 0
        
        for batch in tqdm(train_loader):
            img = batch["masked_img"].cuda()          # img with both masks
            mask = batch["known_and_fake_mask"].cuda()      # real missing gaps and artificial gaps
            mask_aug = batch["fake_mask"].cuda()            # 1=kept, 0=artificial hole
            known_mask = batch['known_mask'].cuda()
            target = batch["target"].cuda()
            p_mask = batch["p_mask"]
            p_val_map = batch["p_val_mask"]

            # If Pandora data exists, move to device
            if p_mask is not None:
                p_mask = p_mask.cuda()
                p_val_map = p_val_map.cuda() if p_val_map is not None else None

            pred_t, _ = model(img, mask_aug)
            
            # === Telea-model fusion ===
            telea_pred = batch.get("telea_pred", None)
            alpha = 0.7   # weight for UNet; 0.7–0.9 typically best
            if telea_pred is not None:
                telea_pred = telea_pred.cuda()
                pred_t = alpha * pred_t + (1 - alpha) * telea_pred
            loss = improved_loss_progress(pred_t, target, mask_aug, epoch, max_epochs=epochs)           
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
            
        
            # Calculate metrics for fake mask regions
            batch_metrics = calculate_metrics(pred_t, target, mask_aug, known_mask, 
                                              p_mask=p_mask, p_values=p_val_map, normalizer=normalizer)

            for key in batch_metrics:
                if key not in train_metrics:
                    train_metrics[key] = 0.0
                train_metrics[key] += batch_metrics[key]

            train_batch_count += 1

        
            
        for key in train_metrics:
            train_metrics[key] /= train_batch_count
        train_loss /= len(train_loader)

        # ---- Validation (Simple Version) ----
        model.eval()
        val_loader.dataset.current_epoch=0
        val_loader.dataset.max_epochs =1
        val_loss = 0
        val_metrics = {
            'pandora_rmse': 0.0, 'pandora_mae': 0.0, 'pandora_rho': 0.0, 'pandora_r2': 0.0,
            'n_pandora_stations': 0
        }
        val_batch_count = 0

        with torch.no_grad():
            for batch in tqdm(val_loader):
                img = batch["masked_img"].cuda()
                mask = batch["known_mask"].cuda()
                target = batch["target"].cuda()
                p_mask = batch["p_mask"]
                p_val_map = batch["p_val_mask"]

                if p_mask is not None:
                    p_mask = p_mask.cuda()
                    p_val_map = p_val_map.cuda() if p_val_map is not None else None

                pred, _ = model(img, mask)
                telea_pred = batch.get("telea_pred", None)
                if telea_pred is not None:
                    telea_pred = telea_pred.cuda()
                    pred = alpha * pred + (1 - alpha) * telea_pred
                loss = improved_loss_progress(pred, target, mask, epoch, max_epochs=epochs)
                val_loss += loss.item()

                # Calculate metrics for first sample in batch only (faster)
                batch_metrics = calculate_metrics(
                    pred, target, mask, mask,  # Use known_mask for both
                    p_mask=p_mask, p_values=p_val_map, normalizer=normalizer,
                )

                # Add to validation totals
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]

                val_batch_count += 1

        # Average validation metrics and loss over all batches
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= val_batch_count

        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"Train metrics - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")   
        
        if train_metrics['n_pandora_stations'] > 0:
            print(f"Train Pandora metrics - RMSE: {train_metrics['pandora_rmse']:.4E}, "
                  f"MAE: {train_metrics['pandora_mae']:.4E}, "
                  f"ρ: {train_metrics['pandora_rho']:.2f}, "
                  f"R²: {train_metrics['pandora_r2']:.2f}")
                  
        if val_metrics['n_pandora_stations'] > 0:
            print(f"Val Pandora metrics - RMSE: {val_metrics['pandora_rmse']:.4E}, "
                  f"MAE: {val_metrics['pandora_mae']:.4E}, "
                  f"ρ: {val_metrics['pandora_rho']:.2f}, "
                  f"R²: {val_metrics['pandora_r2']:.2f}")
        
        # ---- Early stopping ----
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            wait = 0
            torch.save(model.state_dict(), f"pconvunet_{epoch}.pt")
            
        if epoch == epochs - 1:
            torch.save(model.state_dict(), "pconvunet_final.pt")
                
        # Store all metrics in history
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse": train_metrics['rmse'],
            "train_mae": train_metrics['mae'],
            "train_r2": train_metrics['r2'],
            "train_pandora_rmse": train_metrics['pandora_rmse'],
            "train_pandora_mae": train_metrics['pandora_mae'],
            "train_pandora_rho": train_metrics['pandora_rho'],
            "train_pandora_r2": train_metrics['pandora_r2'],
            "train_n_pandora_stations": train_metrics['n_pandora_stations'],
            "val_pandora_rmse": val_metrics['pandora_rmse'],
            "val_pandora_mae": val_metrics['pandora_mae'],
            "val_pandora_rho": val_metrics['pandora_rho'],
            "val_pandora_r2": val_metrics['pandora_r2'],
            "val_n_pandora_stations": val_metrics['n_pandora_stations'],
            "pred_min_range": pred.min().item() if 'pred' in locals() else None,
            "pred_max_range": pred.max().item() if 'pred' in locals() else None
        }
        history.append(epoch_data)
        
        if epoch % 2 == 0: 
            visualize_sample_or_batch(
                    dataloader=val_loader,
                    batch_idx=27, sample_idx=2,
                    model=model,
                    normalizer=normalizer,
                    device="cuda",
                    shp_path=shp_path,
                    pandora_df=pandora_df,
                    inference=True,   # model + stats
                    train=False,
                    station_color_map=station_color_map,
                    save=True,
                    epoch=epoch)
        # Write all metrics to CSV
        fieldnames = [
            "epoch", "train_loss", "val_loss", 
            "train_rmse", "train_mae", "train_r2",
            "train_pandora_rmse", "train_pandora_mae", "train_pandora_rho", "train_pandora_r2", "train_n_pandora_stations",
            "val_pandora_rmse", "val_pandora_mae", "val_pandora_rho", "val_pandora_r2", "val_n_pandora_stations",
            "pred_min_range", "pred_max_range"
        ]
        
        with open('csv_history.csv', "w", newline="") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(history)
        
        
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

