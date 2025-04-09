import glob
import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import os
from VAE import VAE  # Make sure your VAE is defined to produce (reconstructed, mu, logvar)
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class PineappleDataset(Dataset):
    def __init__(self, train=True, train_ratio=0.8):
        # Get all images sorted from the specified folder.
        self.all_images = sorted(glob.glob("./FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED/*"))
        # Calculate the index at which to split the dataset.
        split_index = int(len(self.all_images) * train_ratio)
        # Partition the images based on the 'train' flag.
        if train:
            self.images = self.all_images[:split_index]
        else:
            self.images = self.all_images[split_index:]
        self.resize_shape = (256, 256)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        channels = 3
        # Resize the image.
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # Convert to float32 and normalize.
        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        # Rearrange the dimensions to (channels, height, width).
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
        image = self.transform_image(self.images[idx])
        sample = {'image': image, 'idx': idx}
        return sample

# Define a BCE Loss with 'sum' as per the image snippet
# loss_fn = nn.BCELoss(reduction='sum')

def vae_loss(reconstructed, original, mean, logvar):
    """
    Summation-based BCE + KL divergence, matching the style from your image.

    reconstructed: [batch_size, channels, height, width]
    original: [batch_size, channels, height, width]
    mean, logvar: [batch_size, latent_dim]
    """
    # Flatten for BCE
    b_size = reconstructed.size(0)
    reconstructed = reconstructed.view(b_size, -1)  # [B, D]
    original = original.view(b_size, -1)            # [B, D]

    # If your model outputs raw logits, you should use BCEWithLogitsLoss instead
    # or apply a final sigmoid here:
    reconstructed = torch.sigmoid(reconstructed)

    # 1) Reconstruction loss
    #reconstruction_loss = loss_fn(reconstructed, original)
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction='mean')

    # 2) KL divergence term:
    #    The expression below is the sum over the batch of (1 + log(sigma^2) - mu^2 - sigma^2).
    #    logvar = log(sigma^2)
    KL_DIV = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / b_size

    # Negative ELBO = reconstruction_loss + 0.5 * KL_DIV
    # (the snippet from your image does exactly that).
    return reconstruction_loss + KL_DIV, reconstruction_loss, KL_DIV

def main_1():
    batch_size = 2
    lr = 0.0001
    epochs = 5
    checkpoints_location = "./checkpoints/"
    # Create the checkpoints directory if it doesn't exist
    os.makedirs(checkpoints_location, exist_ok=True)

    # Get API key from environment variable
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    trainset = PineappleDataset(train=True, train_ratio=0.8)
    testset = PineappleDataset(train=False, train_ratio=0.8)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Instantiate the VAE model and move it to GPU.
    vae = VAE().cuda()

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    run = wandb.init(
        entity="imagine_team",
        project="DeepLearning_project_VAE",
        config={
            "learning_rate": lr,
            "architecture": "MLP",
            "dataset": "Pineapples",
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam"
        },
    )
    global_step = 0  # Initialize a global step counter

    # Early stopping and checkpoint parameters
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        with tqdm(total=len(trainset), desc=f'Epoch: {epoch}/{epochs}', unit='img') as prog_bar:
            for i, data in enumerate(trainloader, 0):
                color_images = data["image"].cuda()
                optimizer.zero_grad()

                reconstruction, mean, logvar = vae(color_images)
                loss = vae_loss(reconstruction, color_images, mean, logvar)
                
                loss.backward()
                optimizer.step()

                # Log the loss per training step using the global step counter
                wandb.log({"step_loss": loss.item()}, step=global_step)
                global_step += 1

                prog_bar.set_postfix(loss=loss.item())
                prog_bar.update(color_images.size(0))

                running_loss += loss.item()
                num_batches += 1

        # Compute the average loss for the epoch
        avg_loss = running_loss / num_batches if num_batches else 0
        wandb.log({"epoch_loss": avg_loss}, step=epoch)
        print(f"Epoch {epoch}: Average Loss: {avg_loss}")

        # Check if this epoch improved the best loss so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset the counter as we found an improvement
            checkpoint_name = f"weights_ck_{epoch}.pt"
            torch.save(vae.state_dict(), os.path.join(checkpoints_location, checkpoint_name))
            print(f"Checkpoint saved: {checkpoint_name}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered. No improvement for 5 consecutive epochs.")
            break


def main_2():
    # Basic configuration
    # ---------------------------------------------
    transform = transforms.Compose([
        #transforms.Resize((64, 64)),  # Resize to 256x256
        transforms.ToTensor()
    ])
    dataset_train = CIFAR10(root='./data', download=True, transform=transform)
    dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
    batch_size = 16
    lr = 0.0001
    epochs = 5
    checkpoints_location = "./checkpoints_cifar/"
    os.makedirs(checkpoints_location, exist_ok=True)
    global_step = 0  # Initialize a global step counter
    # Early stopping and checkpoint parameters
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    # ---------------------------------------------

    # Get API key from environment variable
    # ---------------------------------------------
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)

    wandb.init(
        entity="fabian-fallasmoya-universidad-de-costa-rica",
        project="VAE_CIFAR",
        config={
            "learning_rate": lr,
            "architecture": "Base VAE",
            "dataset": "cifar",
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam"
        },
    )
    # ---------------------------------------------
    
    # Model
    vae = VAE().cuda()
    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    # Training
    for epoch in range(10):
        running_loss = 0.0
        num_batches = 0

        with tqdm(total=len(dataset_train), desc=f'Epoch: {epoch}/{epochs}', unit='img') as prog_bar:
            for batch in dataloader_train:
                images, _ = batch
                images = images.cuda()

                optimizer.zero_grad()
                recon_images, mean, logvar = vae(images)
                loss, recon_loss, kl_loss = vae_loss(recon_images, images, mean, logvar)
                loss.backward()
                optimizer.step()

                # record information to be analized
                # ---------------------------------------------
                wandb.log({"step_loss": loss.item()}, step=global_step)
                global_step += 1

                prog_bar.set_postfix(loss=loss.item())
                prog_bar.update(images.size(0))

                running_loss += loss.item()
                num_batches += 1
                # ---------------------------------------------

        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}")

        # save checkpoint if necessary
        # ---------------------------------------------
        avg_loss = running_loss / num_batches if num_batches else 0
        wandb.log({"epoch_loss": avg_loss}, step=epoch)
        print(f"Epoch {epoch}: Average Loss: {avg_loss}")

        # Check if this epoch improved the best loss so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset the counter as we found an improvement
            checkpoint_name = f"weights_ck_{epoch}.pt"
            torch.save(vae.state_dict(), os.path.join(checkpoints_location, checkpoint_name))
            print(f"Checkpoint saved: {checkpoint_name}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
        # ---------------------------------------------
            

        # Early stopping check
        # ---------------------------------------------
        if patience_counter >= patience:
            print("Early stopping triggered. No improvement for 5 consecutive epochs.")
            break
        # ---------------------------------------------

def main_3():
    # Basic configuration
    # ---------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = CIFAR10(root='./data', download=True, transform=transform)

    # Split dataset into train and validation
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.3, random_state=42)
    dataset_train = Subset(dataset, train_indices)
    dataset_val = Subset(dataset, val_indices)

    train_batch_size = 32
    val_batch_size = 32
    dataloader_train = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=val_batch_size, shuffle=False)

    lr = 0.0001
    epochs = 5
    checkpoints_location = "./checkpoints_cifar/"
    os.makedirs(checkpoints_location, exist_ok=True)
    global_step = 0

    best_loss = float('inf')
    patience = 2
    patience_counter = 0

    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    wandb.init(
        entity="fabian-fallasmoya-universidad-de-costa-rica",
        project="VAE_CIFAR",
        config={
            "learning_rate": lr,
            "architecture": "Base VAE",
            "dataset": "cifar",
            "epochs": epochs,
            "batch_size": train_batch_size,
            "optimizer": "Adam"
        },
    )
    # ---------------------------------------------

    vae = VAE().cuda()
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        vae.train()
        running_loss = 0.0
        num_batches = 0

        with tqdm(total=len(dataset_train), desc=f'Epoch: {epoch}/{epochs}', unit='img') as prog_bar:
            for batch in dataloader_train:
                images, _ = batch
                images = images.cuda()

                optimizer.zero_grad()
                recon_images, mean, logvar = vae(images)
                loss, recon_loss, kl_loss = vae_loss(recon_images, images, mean, logvar)
                loss.backward()
                optimizer.step()

                wandb.log({"step_loss": loss.item()}, step=global_step)
                global_step += 1

                prog_bar.set_postfix(loss=loss.item())
                prog_bar.update(images.size(0))

                running_loss += loss.item()
                num_batches += 1

        avg_train_loss = running_loss / num_batches if num_batches else 0
        print(f"Epoch {epoch}: Avg Train Loss={avg_train_loss:.4f}")

        # --- Validation loop ---
        vae.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in dataloader_val:
                val_images, _ = val_batch
                val_images = val_images.cuda()

                recon_images, mean, logvar = vae(val_images)
                loss, _, _ = vae_loss(recon_images, val_images, mean, logvar)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches else float('inf')
        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": avg_val_loss
        }, step=epoch)
        print(f"Epoch {epoch}: Avg Val Loss={avg_val_loss:.4f}")

        # --- Checkpointing ---
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            checkpoint_name = f"weights_ck_{epoch}.pt"
            torch.save(vae.state_dict(), os.path.join(checkpoints_location, checkpoint_name))
            print(f"Checkpoint saved: {checkpoint_name}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main_3()

