import glob
import cv2
import numpy as np
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import wandb
import os
from VAE import VAE  # Make sure your VAE is defined to produce (reconstructed, mu, logvar)

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
loss_fn = nn.BCELoss(reduction='sum')

def vae_loss(reconstructed, original, mean, logvar, loss="bce"):
    """
    Summation-based BCE + KL divergence, matching the style from your image.

    reconstructed: [batch_size, channels, height, width]
    original: [batch_size, channels, height, width]
    mean, logvar: [batch_size, latent_dim]
    """

    b_size = reconstructed.size(0)

    if loss == "bce":
        loss_fn = nn.BCELoss(reduction='sum')
        # Flatten for BCE
        
        reconstructed = reconstructed.view(b_size, -1)  # [B, D]
        original = original.view(b_size, -1)            # [B, D]

        # If your model outputs raw logits, you should use BCEWithLogitsLoss instead
        # or apply a final sigmoid here:
        reconstructed = torch.sigmoid(reconstructed)

        # 1) Reconstruction loss
        reconstruction_loss = loss_fn(reconstructed, original)
    else:
        loss_fn = nn.MSELoss(reduction='sum')
        
        # 1) Reconstruction loss
        reconstruction_loss = loss_fn(reconstructed, original)

    # 2) KL divergence term:
    #    The expression below is the sum over the batch of (1 + log(sigma^2) - mu^2 - sigma^2).
    #    logvar = log(sigma^2)
    KL_DIV = -torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Negative ELBO = reconstruction_loss + 0.5 * KL_DIV
    # (the snippet from your image does exactly that).
    # loss = reconstruction_loss + KL_DIV

    #loss = reconstruction_loss + KL_DIV #0.1 * KL_DIV
    # Less influence for KL at latent space between normal distribution and data distribution
    loss = reconstruction_loss + 0.1 * KL_DIV

    # Often we divide by batch size to get a mean loss per sample
    return loss / b_size

def main():
    batch_size = 4
    lr = 0.0001
    epochs = 50
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
    net = VAE()
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    run = wandb.init(
        entity="deeplearningMG",
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

                reconstruction, mean, logvar = net(color_images)
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
            torch.save(net.state_dict(), os.path.join(checkpoints_location, checkpoint_name))
            print(f"Checkpoint saved: {checkpoint_name}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        # Early stopping check
        if patience_counter >= patience:
            print("Early stopping triggered. No improvement for 5 consecutive epochs.")
            break

if __name__ == "__main__":
    main()