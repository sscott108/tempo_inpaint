import torch
import torch.optim as optim
import csv
from tqdm import tqdm
from .losses import warmup_loss
from ..utils.visualization import visualize_batch

def train_model(model, train_loader, val_loader, epochs=50, patience=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_loss = float("inf")
    best_state = None
    wait = 0

    history =[]
    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader):
            img   = batch["img_w_both_masks"].cuda()          #img with both masks
            mask  = batch["known_and_fake_mask"].cuda()       # real missing gaps and artificial gaps
            mask_aug = batch["fake_mask"].cuda()              # 1=kept, 0=artificial hole
            target= batch["target"].cuda()

            pred_t, pred_mask = model(img, mask)
#             if epoch < 10:
            loss = warmup_loss(pred_t,target,mask)
#             else: loss = criterion(pred_t, target, mask)
            
            
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        visualize_batch(epoch, model, train_ds, idx=300, device="cuda")
        
#         print(f"Target range: {target.min().item():.2f}, {target.max().item():.2f}")
#         print(f"Pred range: {pred.min().item():.2f}, {pred.max().item():.2f}")

            
        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                img   = batch["img_w_both_masks"].cuda()
                mask  = batch["known_and_fake_mask"].cuda()
                target= batch["target"].cuda()
                pred, pred_mask = model(img, mask)
#                 if epoch < 10:
                loss = warmup_loss(pred,target,mask)
#                 else: loss = criterion(pred, target, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)
    
        print(f"Epoch {epoch+1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        
        fill_frac = visualize_batch(epoch, model, val_ds, idx=19, device="cuda")
        
        # ---- Early stopping ----
        if val_loss < best_loss:
            best_loss = val_loss
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
            "pred_min_range": pred_t.min().item(),
            "pred_max_range": pred_t.max().item()
        })
        
        with open('csv_history.csv', "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss", "pred_min_range", "pred_max_range"])
            writer.writeheader()
            writer.writerows(history)
        
    # Restore best weights
    model.load_state_dict(best_state)
    return model