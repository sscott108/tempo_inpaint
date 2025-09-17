import torch
import torch.optim as optim
import csv
from tqdm import tqdm
from ..losses.losses import warmup_loss, calculate_metrics
from ..utils.helpers import visualize_batch

# def train_model(model, normalizer,train_loader, val_loader, shp_path, epochs=50, patience=5):
#     opt = torch.optim.Adam(model.parameters(), lr=1e-5)
#     best_loss = float("inf")
#     best_state = None
    
#     wait = 0

#     history = []
#     for epoch in range(epochs):
#         # ---- Training ----
#         model.train()
#         train_loss = 0
#         train_metrics = {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
#         batch_count = 0
        
#         for batch in tqdm(train_loader):
#             img = batch["masked_img"].cuda()          # img with both masks
#             mask = batch["known_and_fake_mask"].cuda()      # real missing gaps and artificial gaps
#             mask_aug = batch["fake_mask"].cuda()            # 1=kept, 0=artificial hole
#             known_mask = batch['known_mask'].cuda()
#             target = batch["target"].cuda()

#             pred_t, _ = model(img, mask_aug)
#             loss = warmup_loss(pred_t, target, mask_aug)            
            
#             opt.zero_grad(); loss.backward(); opt.step()
#             train_loss += loss.item()
            
#             # Calculate metrics for fake mask regions
#             batch_metrics = calculate_metrics(pred_t, target, mask_aug, known_mask)
#             for key in batch_metrics:
#                 train_metrics[key] += batch_metrics[key]
#             batch_count += 1
#         visualize_batch(epoch, model,normalizer, train_loader, device="cuda",train=True, shp_path=shp_path,save=False)

#         # Average metrics over batches
#         train_loss /= len(train_loader)
#         for key in train_metrics:
#             train_metrics[key] /= batch_count

#         # ---- Validation ----
#         model.eval()
#         val_loss = 0
#         batch_count = 0
        
#         with torch.no_grad():
#             for batch in tqdm(val_loader):
#                 img = batch["masked_img"].cuda()
#                 mask = batch["known_mask"].cuda()
#                 target = batch["target"].cuda()
#                 pred, _ = model(img, mask)

#                 loss = warmup_loss(pred, target, mask)
#                 val_loss += loss.item()
                
#                 # Calculate metrics for fake mask regions
#                 batch_count += 1
                
#         val_loss /= len(val_loader)
       
#         visualize_batch(epoch, model, normalizer, val_loader, device="cuda", train=False, shp_path=shp_path,save = False)
#         print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
#         print(f"Train metrics - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")   
        
        
#         # ---- Early stopping ----
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_state = model.state_dict().copy()
#             wait = 0
#             torch.save(model.state_dict(), "pconvunet.pt")
#         else:
#             wait += 1
#             if wait >= patience:
#                 print(f"Early stopping at epoch {epoch+1}")
#                 break
                
#         history.append({
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "val_loss": val_loss,
#             "train_rmse": train_metrics['rmse'],
#             "train_mae": train_metrics['mae'],
#             "train_r2": train_metrics['r2'],
#             "pred_min_range": pred.min().item() if 'pred' in locals() else None,
#             "pred_max_range": pred.max().item() if 'pred' in locals() else None
#         })
        
#         with open('csv_history.csv', "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=[
#                 "epoch", "train_loss", "val_loss", 
#                 "train_rmse", "train_mae", "train_r2",
#                 "pred_min_range", "pred_max_range"
#             ])
#             writer.writeheader()
#             writer.writerows(history)
        
#     # Restore best weights
#     if best_state is not None:
#         model.load_state_dict(best_state)
#     return model
def train_model(model, normalizer,train_loader, val_loader, shp_path, epochs=50, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_loss = float("inf")
    best_state = None
    
    wait = 0

    history = []
    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0
        train_metrics = {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
        batch_count = 0
        
        for batch in tqdm(train_loader):
            img = batch["masked_img"].cuda()          # img with both masks
            mask = batch["known_and_fake_mask"].cuda()      # real missing gaps and artificial gaps
            mask_aug = batch["fake_mask"].cuda()            # 1=kept, 0=artificial hole
            known_mask = batch['known_mask'].cuda()
            target = batch["target"].cuda()
            p_mask = batch.get("p_mask", None)
            p_val_map = batch.get("p_val_mask", None)

            # If Pandora data exists, move to device
            if p_mask is not None:
                p_mask = p_mask.cuda()
                p_val_map = p_val_map.cuda() if p_val_map is not None else None

            pred_t, _ = model(img, mask_aug)
            loss = warmup_loss(pred_t, target, mask_aug)            
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
            
            # Calculate metrics for fake mask regions
            batch_metrics = calculate_metrics(
                pred_t, target, mask_aug, known_mask, 
                p_mask=p_mask, p_values=p_val_map, normalizer=normalizer)

            for key in batch_metrics:
                if key not in train_metrics:
                    train_metrics[key] = 0.0
                train_metrics[key] += batch_metrics[key]

            batch_count += 1
            for key in train_metrics:
                train_metrics[key] /= batch_count
                
#             visualize_batch(epoch, model,normalizer, train_loader, device="cuda",train=True, shp_path=shp_path,save=False)

#             visualize_batch(epoch, model, normalizer, val_loader, device="cuda", train=False, shp_path=shp_path,save = False)
        # Average metrics over batches
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= batch_count

        # ---- Validation ----
        model.eval()
        val_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                img = batch["masked_img"].cuda()
                mask = batch["known_mask"].cuda()
                target = batch["target"].cuda()
                p_mask = batch.get("p_mask", None)
                p_val_map = batch.get("p_val_mask", None)

                if p_mask is not None:
                    p_mask = p_mask.cuda()
                    p_val_map = p_val_map.cuda() if p_val_map is not None else None
                    
                pred, _ = model(img, mask)
                loss = warmup_loss(pred, target, mask)
                val_loss += loss.item()
                
                # Calculate metrics for fake mask regions
                batch_metrics = calculate_metrics(
                pred_t, target, mask_aug, known_mask, 
                p_mask=p_mask, p_values=p_val_map, normalizer=normalizer)

                batch_metrics = calculate_metrics(pred, target, mask)
                for key in batch_metrics:
                    val_metrics[key] += batch_metrics[key]
                batch_count += 1
                
        val_loss /= len(val_loader)
       
#         visualize_batch(epoch, model, normalizer, val_loader, device="cuda", train=False, shp_path=shp_path,save = False)
        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        print(f"Train metrics - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")   
        print(f"Train metrics - RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        
        if train_metrics['n_pandora_stations'] > 0:
            print(f"Train Pandora metrics - RMSE: {train_metrics['pandora_rmse']:.4E}, "
                  f"MAE: {train_metrics['pandora_mae']:.4E}, "
                  f"ρ: {train_metrics['pandora_rho']:.2f}, "
                  f"R²: {train_metrics['pandora_r2']:.2f}")
        
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
                
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse": train_metrics['rmse'],
            "train_mae": train_metrics['mae'],
            "train_r2": train_metrics['r2'],
            "pred_min_range": pred.min().item() if 'pred' in locals() else None,
            "pred_max_range": pred.max().item() if 'pred' in locals() else None
        })
        
        with open('csv_history.csv', "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "val_loss", 
                "train_rmse", "train_mae", "train_r2",
                "pred_min_range", "pred_max_range"
            ])
            writer.writeheader()
            writer.writerows(history)
        
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
