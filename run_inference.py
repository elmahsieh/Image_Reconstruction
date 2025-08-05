import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_msssim import ssim
import torch.nn.functional as F
import math

# ---- Load Your Trained Model ----
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())

        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.final = nn.Sequential(nn.Conv2d(128, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(e3)
        d2_cat = torch.cat([d2, e2], dim=1)
        d1 = self.dec1(d2_cat)
        d1_cat = torch.cat([d1, e1], dim=1)
        return self.final(d1_cat)

# ---- Device Setup ----
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Using MPS (Apple GPU)")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚠️ Using fallback device: {DEVICE}")

# ---- Load Trained Weights ----
model = UNetAutoencoder().to(DEVICE)
model.load_state_dict(torch.load("best_inpainting_model.pt", map_location=DEVICE))
model.eval()

# ---- Transforms ----
transform_rgb = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# ---- Preprocessing Functions ----
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    return transform_rgb(img)

def preprocess_with_mask(img_path, mask_path):
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img_tensor = transform_rgb(img)
    mask_tensor = transform_mask(mask)
    mask_tensor = (mask_tensor > 0.5).float()  # Binarize to 0 or 1

    masked_img = img_tensor * mask_tensor
    return masked_img, img_tensor

def apply_random_mask(img_tensor):
    masked = img_tensor.clone()
    mask = torch.ones_like(masked)
    h, w = masked.shape[1], masked.shape[2]
    mask_size = np.random.randint(8, 16)
    top = np.random.randint(0, h - mask_size)
    left = np.random.randint(0, w - mask_size)
    mask[:, top:top+mask_size, left:left+mask_size] = 0
    return masked * mask

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # since your images are normalized between 0 and 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_ssim(img1, img2):
    # img1, img2 should be torch tensors with shape [C,H,W] and pixel values 0-1
    return ssim(img1.unsqueeze(0), img2.unsqueeze(0), data_range=1.0, size_average=True).item()

# ---- Inference and Visualization ----
def run_and_show(disrupted_img_path, mask_path=None, original_path=None):
    # Load and preprocess original image if provided
    if original_path:
        original = Image.open(original_path).convert("RGB")
        original = transform_rgb(original)
    else:
        original = None  # Or raise an error if you want original mandatory

    disrupted = Image.open(disrupted_img_path).convert("L")
    disrupted = transform_mask(disrupted).repeat(3, 1, 1)  # Convert to 3-channel
    input_tensor = disrupted.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        restored = model(input_tensor).squeeze(0).cpu()

    # Calculate PSNR and SSIM only if original is provided
    if original is not None:
        psnr_val = calculate_psnr(original, restored)
        ssim_val = calculate_ssim(original, restored)
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"SSIM: {ssim_val:.4f}")

    def to_numpy(tensor):
        return tensor.permute(1, 2, 0).numpy()

    # Prepare images for plotting
    imgs = [("Disrupted Input", disrupted)]
    if original is not None:
        imgs.insert(0, ("Original", original))
    imgs.append(("Restored Output", restored))

    if mask_path:
        mask = Image.open(mask_path).convert("L")
        mask_tensor = transform_mask(mask)
        imgs.insert(-1, ("Mask", mask_tensor.repeat(3, 1, 1)))  # Optional

    fig, axs = plt.subplots(1, len(imgs), figsize=(4 * len(imgs), 4))
    for ax, (title, img_tensor) in zip(axs, imgs):
        ax.imshow(to_numpy(img_tensor))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    

# ---- Run It ----
if __name__ == "__main__":

    run_and_show(
        disrupted_img_path="img/test_1.png",
        mask_path="img/test_1_mask.png",
        original_path="img/smallTriangle.png"  # Optional, for comparison
    )

    # run_and_show(
    #     disrupted_img_path="img/test_2_disrupted.png",
    #     mask_path="img/test_2_binary.png",
    #     original_path="img/test_2_image.png"  # Optional, for comparison
    # )

