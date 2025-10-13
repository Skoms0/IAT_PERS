from dataclasses import dataclass, field
import torch
import os, json

@dataclass
class MedNISTConfig:
    # ===== General =====
    seed: int = 42
    data_dir: str = "./data/MedNIST"
    download: bool = True
    save_dir: str = "./results_hands_opt"

    # ===== Data =====
    image_size: int = 128        # Higher resolution improves detail replication
    min_intensity: float = 0.0
    max_intensity: float = 1.0
    train_valid_ratio: float = 0.9
    batch_size: int = 64         # Smaller batch = more gradient precision
    num_workers: int = 4
    persistent_workers: bool = True

    # ===== Model architecture =====
    latent_dim: int = 256        # Larger latent space preserves fine structures
    encoder_channels: list = field(default_factory=lambda: [1, 64, 128, 256, 512, 512])
    decoder_channels: list = field(default_factory=lambda: [512, 512, 256, 128, 64, 1])
    kernel_size: int = 3
    stride: int = 2
    padding: int = 1
    activation: str = "LeakyReLU"
    output_activation: str = "Sigmoid"
    dropout_rate: float = 0.1    # Lower dropout improves detail retention
    use_batchnorm: bool = True

    # ===== Training =====
    num_epochs: int = 80         # Longer training for better convergence
    learning_rate: float = 2e-4  # Slightly lower for stable reconstruction
    weight_decay: float = 1e-6
    loss_function: str = "MSELoss"  # You can optionally test "L1Loss" for sharper recon
    scheduler: str = "CosineAnnealingLR"

    # ===== Misc =====
    labels: list = field(default_factory=lambda: [
        'AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT'
    ])
    selected_label: str = "Hand"
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # ===== Utility =====
    def summary(self):
        print("===== Optimized MedNIST Autoencoder Configuration (Hands) =====")
        for key, value in self.__dict__.items():
            print(f"{key:20}: {value}")
        print("===============================================================")

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(self.save_dir, "config.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=4, default=str)
        print(f"[✔] Configuration saved to {filepath}")
