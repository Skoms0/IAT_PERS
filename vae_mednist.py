"""
vae_mednist.py
Convolutional Variational Autoencoder (VAE) for MedNIST images.
Features:
- CNN encoder/decoder
- Reparameterization trick
- Reconstruction + KL loss
- AdamW optimizer + Cosine LR scheduler
- Early stopping & checkpointing
"""

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

train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)


# ======================================================
# Model Definition
# ======================================================
class ConvVAE(nn.Module):
    def __init__(self, cfg: MedNISTConfig):
        super().__init__()
        act_fn = getattr(nn, cfg.activation)

        # --- Encoder ---
        enc_layers = []
        for in_c, out_c in zip(cfg.encoder_channels[:-1], cfg.encoder_channels[1:]):
            enc_layers.append(nn.Conv2d(in_c, out_c, cfg.kernel_size, cfg.stride, cfg.padding))
            enc_layers.append(nn.BatchNorm2d(out_c))
            enc_layers.append(act_fn(inplace=True))
        self.encoder = nn.Sequential(*enc_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, cfg.image_size, cfg.image_size)
            out = self.encoder(dummy)
            self.enc_shape = out.shape[1:]
            self.enc_flat = int(np.prod(self.enc_shape))

        # Latent space
        self.fc_mu = nn.Linear(self.enc_flat, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat, cfg.latent_dim)
        self.fc_decode = nn.Linear(cfg.latent_dim, self.enc_flat)

        # --- Decoder ---
        dec_layers = []
        for in_c, out_c in zip(cfg.decoder_channels[:-1], cfg.decoder_channels[1:]):
            dec_layers.append(nn.ConvTranspose2d(in_c, out_c, cfg.kernel_size, cfg.stride, cfg.padding))
            if out_c != 1:
                dec_layers.append(nn.BatchNorm2d(out_c))
                dec_layers.append(act_fn(inplace=True))
            else:
                dec_layers.append(nn.Sigmoid())  # Output in [0,1]
        self.decoder = nn.Sequential(*dec_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        orig_size = x.shape[-2:]
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        x = self.fc_decode(z)
        x = x.view(x.size(0), *self.enc_shape)
        x = self.decoder(x)

        if x.shape[-2:] != orig_size:
            x = nn.functional.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)

        return x, mu, logvar


# ======================================================
# Loss Function
# ======================================================
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + cfg.beta * kld) / x.size(0)


# ======================================================
# Training Setup
# ======================================================
device = cfg.device
model = ConvVAE(cfg).to(device)
optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

best_val_loss = np.inf
epochs_no_improve = 0
patience = 10
checkpoint_path = os.path.join(cfg.save_dir, "best_vae.pth")

train_losses, valid_losses = [], []

print("\n🚀 Starting VAE training...\n")
for epoch in range(cfg.num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}"):
        imgs = batch["image"].to(device).float()
        optimizer.zero_grad()
        outputs, mu, logvar = model(imgs)
        loss = vae_loss(outputs, imgs, mu, logvar)
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
            outputs, mu, logvar = model(imgs)
            loss = vae_loss(outputs, imgs, mu, logvar)
            val_loss += loss.item() * imgs.size(0)
    val_loss /= len(valid_loader.dataset)
    scheduler.step()

    train_losses.append(train_loss)
    valid_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{cfg.num_epochs}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[✔] Validation improved. Saved model to {checkpoint_path}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\n⏹ Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
            break


# ======================================================
# Visualization
# ======================================================
plt.figure(figsize=(6,4))
plt.plot(train_losses, label='Train')
plt.plot(valid_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('VAE Loss')
plt.legend()
plt.title('VAE Training Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.save_dir, "vae_training_curve.png"))
plt.close()


def show_reconstructions(model, loader, n=8):
    model.eval()
    imgs = next(iter(loader))["image"][:n].to(device)
    with torch.no_grad():
        recs, _, _ = model(imgs)
    imgs, recs = imgs.cpu().numpy(), recs.cpu().numpy()
    fig, axes = plt.subplots(2, n, figsize=(15, 4))
    for i in range(n):
        axes[0, i].imshow(np.squeeze(imgs[i]), cmap='gray')
        axes[1, i].imshow(np.squeeze(recs[i]), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original")
            axes[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "vae_reconstructions.png"))
    plt.close()


print("\n🔍 Loading best VAE and generating reconstructions...")
model.load_state_dict(torch.load(checkpoint_path))
show_reconstructions(model, test_loader)
print("\n✅ VAE training complete. Best validation loss:", round(best_val_loss, 6))
