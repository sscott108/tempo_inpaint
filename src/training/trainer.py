import torch
import torch.optim as optim
import csv
from tqdm import tqdm
from ..losses.losses import improved_loss, calculate_metrics
from ..utils.helpers import visualize_batch
import csv

def train_model(model, normalizer, train_loader, val_loader, shp_path, epochs=50, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_loss = float("inf")
    best_state = None
    wait = 0
    history = []
    
    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0
        train_metrics = {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0,'pandora_rmse': 0.0, 'pandora_mae': 0.0, 'pandora_rho': 0.0, 'pandora_r2': 0.0,'n_pandora_stations': 0}
        train_batch_count = 0
        
        for batch in (train_loader):
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
            loss = improved_loss(pred_t, target, mask_aug)            
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
#             visualize_batch(epoch=epoch, model=model, normalizer=normalizer, dataloader=train_loader,
#                     batch_idx=0, sample_idx=0, device="cuda", save=False, train=True, shp_path=shp_path)
            
#             visualize_batch(epoch=epoch, model=model, normalizer=normalizer, dataloader=val_loader,
#                     batch_idx=0, sample_idx=0, device="cuda", save=False, train=False, shp_path=shp_path)
        
            
        for key in train_metrics:
            train_metrics[key] /= train_batch_count
        train_loss /= len(train_loader)

        # ---- Validation (Simple Version) ----
        model.eval()
        val_loss = 0
        val_metrics = {
            'pandora_rmse': 0.0, 'pandora_mae': 0.0, 'pandora_rho': 0.0, 'pandora_r2': 0.0,
            'n_pandora_stations': 0
        }
        val_batch_count = 0

        with torch.no_grad():
            for batch in (val_loader):
                img = batch["masked_img"].cuda()
                mask = batch["known_mask"].cuda()
                target = batch["target"].cuda()
                p_mask = batch["p_mask"]
                p_val_map = batch["p_val_mask"]

                if p_mask is not None:
                    p_mask = p_mask.cuda()
                    p_val_map = p_val_map.cuda() if p_val_map is not None else None

                pred, _ = model(img, mask)
                loss = improved_loss(pred, target, mask)
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
            torch.save(model.state_dict(), "pconvunet.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
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
            visualize_batch(epoch=epoch, model=model, normalizer=normalizer, dataloader=train_loader,
                    batch_idx=0, sample_idx=0, device="cuda", save=True, train=True, shp_path=shp_path)
            
            visualize_batch(epoch=epoch, model=model, normalizer=normalizer, dataloader=val_loader,
                    batch_idx=0, sample_idx=0, device="cuda", save=True, train=False, shp_path=shp_path)
        
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