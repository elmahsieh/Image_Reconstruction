# train_denoising_autoencoder.py
# train_inpainting_autoencoder.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim  # install via: pip install pytorch-msssim

# ---- Config ----
BATCH_SIZE = 128
EPOCHS = 50
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using MPS (Apple GPU)")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚠️ Using fallback device: {DEVICE}")

PATIENCE = 5

# ---- Masked CIFAR-10 Dataset ----
class MaskedCIFAR10(Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        mask = torch.ones_like(img)

        # Mask out random square region
        h, w = img.shape[1], img.shape[2]
        mask_size = np.random.randint(8, 16)
        top = np.random.randint(0, h - mask_size)
        left = np.random.randint(0, w - mask_size)
        mask[:, top:top + mask_size, left:left + mask_size] = 0

        damaged_img = img * mask
        return damaged_img, img

# ---- Data Loaders ----
full_train = MaskedCIFAR10(train=True)
train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_data, val_data = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# ---- U-Net-like Autoencoder ----
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())  # Down
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())  # Down

        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.final = nn.Sequential(nn.Conv2d(128, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)         # [B, 64, 32, 32]
        e2 = self.enc2(e1)        # [B, 128, 16, 16]
        e3 = self.enc3(e2)        # [B, 256, 8, 8]

        d2 = self.dec2(e3)        # [B, 128, 16, 16]
        d2_cat = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2_cat)    # [B, 64, 32, 32]
        d1_cat = torch.cat([d1, e1], dim=1)

        out = self.final(d1_cat)
        return out

# ---- Loss Function: MSE + SSIM ----
def combined_loss(output, target):
    mse = nn.functional.mse_loss(output, target)
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)
    return mse + 0.2 * ssim_loss  # Adjust weight based on experiments

# ---- Model, Optimizer, Scheduler ----
model = UNetAutoencoder().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

# ---- Training Loop with Early Stopping ----
train_losses = []
val_losses = []
best_val_loss = float('inf')
counter = 0

print("Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for damaged, clean in train_loader:
        damaged, clean = damaged.to(DEVICE), clean.to(DEVICE)
        output = model(damaged)
        loss = combined_loss(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for damaged, clean in val_loader:
            damaged, clean = damaged.to(DEVICE), clean.to(DEVICE)
            output = model(damaged)
            val_loss = combined_loss(output, clean)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_inpainting_model.pt')
    else:
        counter += 1
        if counter >= PATIENCE:
            print("⛔ Early stopping triggered.")
            break

# ---- Plot Losses ----
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()



'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# ---- Config ----
BATCH_SIZE = 128
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Noisy CIFAR-10 Dataset ----
class NoisyCIFAR10(Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.CIFAR10(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        noisy_img = img + 0.5 * torch.randn_like(img)
        noisy_img = torch.clip(noisy_img, 0., 1.)
        return noisy_img, img

# ---- Load and Split Data ----
full_train = NoisyCIFAR10(train=True)
train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_data, val_data = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model Definition ----
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ---- Initialize Model ----
model = DenoisingAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- Training with Early Stopping ----
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 3
counter = 0

print("Training started with early stopping...")
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for noisy_imgs, clean_imgs in train_loader:
        noisy_imgs, clean_imgs = noisy_imgs.to(DEVICE), clean_imgs.to(DEVICE)

        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for noisy_imgs, clean_imgs in val_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(DEVICE), clean_imgs.to(DEVICE)
            outputs = model(noisy_imgs)
            val_loss = criterion(outputs, clean_imgs)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_denoising_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print("⛔ Early stopping triggered!")
            break

# ---- Plot Loss Curve ----
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
'''