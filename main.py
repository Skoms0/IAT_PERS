import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from monai.apps import MedNISTDataset
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, Resized
from parameters import MedNISTConfig


# ======================================================
# Configuration
# ======================================================
cfg = MedNISTConfig()
os.makedirs(cfg.save_dir, exist_ok=True)
cfg.summary()
cfg.save()


# ======================================================
# Data Preparation
# ======================================================
train_set = MedNISTDataset(root_dir=cfg.data_dir, section="training", download=cfg.download, seed=cfg.seed)
test_set = MedNISTDataset(root_dir=cfg.data_dir, section="validation", download=cfg.download, seed=cfg.seed)

train_datalist = [{"image": item["image"], "label": cfg.selected_label}
                  for item in train_set.data if item["class_name"] == cfg.selected_label]
test_datalist = [{"image": item["image"], "label": cfg.selected_label}
                 for item in test_set.data if item["class_name"] == cfg.selected_label]

all_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0,
                         b_min=cfg.min_intensity, b_max=cfg.max_intensity, clip=True),
    Resized(keys=["image"], spatial_size=[cfg.image_size, cfg.image_size]),
])

train_dataset = Dataset(data=train_datalist, transform=all_transforms)
test_data = Dataset(data=test_datalist, transform=all_transforms)

train_size = int(cfg.train_valid_ratio * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_data, valid_data = random_split(train_dataset, [train_size, valid_size])

train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, persistent_workers=cfg.persistent_workers)
valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, persistent_workers=cfg.persistent_workers)
test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, persistent_workers=cfg.persistent_workers)


# ======================================================
# Model Definition
# ======================================================
class ConvAutoencoder(nn.Module):
    def __init__(self, cfg: MedNISTConfig):
        super().__init__()
        act_fn = getattr(nn, cfg.activation)

        # --- Encoder ---
        enc_layers = []
        for in_c, out_c in zip(cfg.encoder_channels[:-1], cfg.encoder_channels[1:]):
            enc_layers.append(nn.Conv2d(in_c, out_c, cfg.kernel_size, cfg.stride, cfg.padding))
            if cfg.use_batchnorm:
                enc_layers.append(nn.BatchNorm2d(out_c))
            enc_layers.append(act_fn(inplace=True))
            enc_layers.append(nn.Dropout2d(cfg.dropout_rate))
        self.encoder = nn.Sequential(*enc_layers)

        # Compute final encoder output shape dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.image_size, cfg.image_size)
            out = self.encoder(dummy)
            self.enc_shape = out.shape[1:]  # (C, H, W)
            self.enc_flat = int(np.prod(self.enc_shape))

        # --- Latent space ---
        self.fc1 = nn.Linear(self.enc_flat, cfg.latent_dim)
        self.fc2 = nn.Linear(cfg.latent_dim, self.enc_flat)

        # --- Decoder ---
        dec_layers = []
        for in_c, out_c in zip(cfg.decoder_channels[:-1], cfg.decoder_channels[1:]):
            dec_layers.append(nn.ConvTranspose2d(in_c, out_c, cfg.kernel_size, cfg.stride, cfg.padding))
            if out_c != 1:
                if cfg.use_batchnorm:
                    dec_layers.append(nn.BatchNorm2d(out_c))
                dec_layers.append(act_fn(inplace=True))
                dec_layers.append(nn.Dropout2d(cfg.dropout_rate))
            else:
                dec_layers.append(getattr(nn, cfg.output_activation)())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        orig_size = x.shape[-2:]  # (H, W)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc1(x)
        x = self.fc2(z)
        x = x.view(x.size(0), *self.enc_shape)
        x = self.decoder(x)

        # Force final output to match input spatial size
        if x.shape[-2:] != orig_size:
            x = nn.functional.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        return x



# ======================================================
# Training Setup
# ======================================================
device = cfg.device
model = ConvAutoencoder(cfg).to(device)
criterion = getattr(nn, cfg.loss_function)()
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

# Early stopping parameters
patience = 10
best_val_loss = np.inf
epochs_no_improve = 0
checkpoint_path = os.path.join(cfg.save_dir, "best_model.pth")


# ======================================================
# Training Loop
# ======================================================
train_losses, valid_losses = [], []

print("\n Starting training...\n")
for epoch in range(cfg.num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
        imgs = batch["image"].to(device).float()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            imgs = batch["image"].to(device).float()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(valid_loader.dataset)
    scheduler.step()

    train_losses.append(train_loss)
    valid_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{cfg.num_epochs}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

    # --- Early Stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[✔] Validation loss improved. Model saved to {checkpoint_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\n⏹ Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
            break


# ======================================================
# Plot Training Curve
# ======================================================
plt.figure(figsize=(6,4))
plt.plot(train_losses, label='Train')
plt.plot(valid_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Autoencoder Training Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.save_dir, "training_curve.png"))
plt.close()
print(f"[✔] Training curve saved to {cfg.save_dir}/training_curve.png")


# ======================================================
# Reconstruction Visualization
# ======================================================
def show_reconstructions(model, loader, n=8):
    model.eval()
    imgs = next(iter(loader))["image"][:n].to(device)
    with torch.no_grad():
        recs = model(imgs)
    imgs, recs = imgs.cpu().numpy(), recs.cpu().numpy()

    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axes[0, i].imshow(np.squeeze(imgs[i]), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original")

        axes[1, i].imshow(np.squeeze(recs[i]), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "reconstructions.png"))
    plt.close()
    print(f"[✔] Reconstructions saved to {cfg.save_dir}/reconstructions.png")


# ======================================================
# Load Best Model & Evaluate
# ======================================================
print("\nLoading best model and generating reconstructions...")
model.load_state_dict(torch.load(checkpoint_path))
show_reconstructions(model, test_loader)
print("\nTraining complete. Best validation loss:", round(best_val_loss, 6))
