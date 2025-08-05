# Image Denoising Autoencoder

## Project Overview

This project implements a convolutional denoising autoencoder using PyTorch to reconstruct clean images from noisy inputs. Using the CIFAR-10 dataset, the model learns to remove Gaussian noise added to images, effectively performing image reconstruction.

## Features

- Uses CIFAR-10 dataset with synthetic Gaussian noise added
- Convolutional autoencoder architecture with encoder and decoder modules
- Trains with Mean Squared Error loss and Adam optimizer
- Early stopping implemented to prevent overfitting
- Saves best model checkpoint (`best_denoising_model.pt`)
- Automatically uses GPU if available, otherwise CPU
- Visualizes training and validation loss curves

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

Install dependencies via:

```bash
pip install requirements.txt
```

## Usage

### Training
Run the training script:

```bash
python train_denoising_autoencoder.py
```

The model will download CIFAR-10 automatically, train with noisy images, and save the best model checkpoint.

Testing / Inference
Use the saved model checkpoint best_denoising_model.pt to denoise your own images. (See test_denoising_autoencoder.py for example usage.)

How It Works (Explanation for Supervisor)

This project implements a denoising autoencoder using PyTorch to reconstruct clean images from noisy inputs. Here's a step-by-step explanation:

1. Data Preparation
We use the CIFAR-10 dataset (60,000 32x32 color images of 10 classes).
We add Gaussian noise to each image by adding random normal noise scaled by 0.5.
The dataset returns pairs of (noisy_image, clean_image) for supervised training.
The dataset is split into 90% training and 10% validation sets to monitor performance during training.
2. Model Architecture
The denoising autoencoder consists of two parts:
Encoder: Compresses the image from 3 channels to a lower-dimensional representation using 3 convolutional layers with ReLU activations.
Decoder: Reconstructs the clean image back from the encoded representation using 3 transpose convolutional layers, ending with a sigmoid activation to ensure outputs between 0 and 1.
3. Training Procedure
The model is trained with Mean Squared Error (MSE) loss, which measures pixel-wise difference between the output and the original clean image.
The Adam optimizer updates the model weights.
We train for up to 50 epochs but use early stopping to avoid overfitting:
If validation loss does not improve for 3 consecutive epochs (patience=3), training stops early.
The best model (lowest validation loss) is saved as best_denoising_model.pt.
4. Performance Monitoring
Training and validation losses are logged each epoch.
At the end, a plot of training vs validation loss is shown to visualize the learning progress.
5. Device Handling
The model automatically runs on GPU if available, otherwise CPU.

### Summary:
The autoencoder learns to remove noise from images by encoding and decoding them, using clean images as a ground truth. Early stopping prevents overtraining, and validation monitoring ensures the model generalizes well.



project-folder/
│
├── train_denoising_autoencoder.py
├── denoise_custom_image.py
├── best_denoising_model.pt  ← saved after training
├── your_image.jpg           ← replace with your test image
