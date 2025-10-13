"""
parameters.py

Optimized configuration for MedNIST Autoencoder.
Improves reconstruction quality and training stability.
"""

from dataclasses import dataclass, field
import torch
import json
import os


@dataclass
class MedNISTConfig:
    # ===== General =====
    seed: int = 42
    data_dir: str = "./data/MedNIST"
    download: bool = True
    save_dir: str = "./results_best"

    # ===== Data =====
    image_size: int = 64
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    train_valid_ratio: float = 0.85
    batch_size: int = 128
    num_workers: int = 4
    persistent_workers: bool = True

    # ===== Model architecture =====
    latent_dim: int = 128
    encoder_channels: list = field(default_factory=lambda: [1, 32, 64, 128, 256, 512])
    decoder_channels: list = field(default_factory=lambda: [512, 256, 128, 64, 32, 1])
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    activation: str = "LeakyReLU"
    output_activation: str = "Sigmoid"

    # ===== Training =====
    num_epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    loss_function: str = "MSELoss"
    scheduler: str = "CosineAnnealingLR"
    dropout_rate: float = 0.2
    use_batchnorm: bool = True

    # ===== Misc =====
    labels: list = field(default_factory=lambda: [
        'AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT'
    ])
    selected_label: str = "Hand"
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ===== Utility =====
    def summary(self):
        print("===== Optimized MedNIST Autoencoder Configuration =====")
        for key, value in self.__dict__.items():
            print(f"{key:20}: {value}")
        print("========================================================")

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(self.save_dir, "config.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=4, default=str)
        print(f"[✔] Configuration saved to {filepath}")
