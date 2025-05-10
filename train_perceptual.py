import os
import torch
import wandb
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import PineappleDataset 
from VAE import VAE, vae_loss
from utils import setup_wandb, parse_args, create_directory
from perceptual_loss import build_vgg_perceptual_model, vae_perceptual_loss

def train_vae(args):

    # Create the folder for checkpoints
    path_to_save_checkpoints = os.path.join(args.checkpoints, f"betaKL@{args.beta_kl_loss}")
    create_directory(path_to_save_checkpoints)

    # Set up the wandb 
    setup_wandb(args)

    # Load data
    trainset = PineappleDataset(train=True, train_ratio=args.train_ratio, path=args.dataset, resize_img=224)
    testset = PineappleDataset(train=False, train_ratio=args.train_ratio, path=args.dataset, resize_img=224)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Perceptual loss models
    layers = ['relu_2', 'relu_3', 'relu_5', 'relu_14', 'relu_15', 'relu_16'] #['relu_13', 'relu_14', 'relu_15', 'relu_16']
    perceptual_model = build_vgg_perceptual_model(layers)

    model_vae = VAE().to(args.device)
    optimizer = optim.Adam(model_vae.parameters(), lr=args.lr)

    global_step = 0
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):

        model_vae.train()

        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kl_loss = 0.0
        running_content_loss = 0.0
        running_style_loss = 0.0

        num_batches = 0

        with tqdm(total=len(trainset), desc=f'Epoch: {epoch}/{args.epochs}', unit='img') as prog_bar:
            for i, data in enumerate(trainloader):
                color_images = data["image"].cuda()
                optimizer.zero_grad()

                reconstruction, mean, logvar = model_vae(color_images)
                #loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, color_images, mean, logvar, kl_beta=args.beta_kl_loss)
                loss, reconstruction_loss, content_loss, style_loss, kl_loss = vae_perceptual_loss(reconstruction,  
                                                                                             color_images, mean, logvar, perceptual_model, 
                                                                                             content_layers=['relu_2', 'relu_3', 'relu_5'],
                                                                                             kl_beta=args.beta_kl_loss,
                                                                                             style_weight=args.style_weight, content_weight=args.content_weight)

                loss.backward()
                optimizer.step()

                global_step += 1

                prog_bar.set_postfix(loss=loss.item())
                prog_bar.update(color_images.size(0))

                running_loss += loss.item()
                running_reconstruction_loss += reconstruction_loss.item()
                running_kl_loss += kl_loss.item()
                running_content_loss += content_loss.item()
                running_style_loss += style_loss.item()

                num_batches += 1
    
        avg_loss = running_loss / num_batches if num_batches else 0
        avg_recon_loss = running_reconstruction_loss / num_batches if num_batches else 0
        avg_kl_loss = running_kl_loss / num_batches if num_batches else 0
        avg_content_loss = running_content_loss / num_batches if num_batches else 0
        avg_style_loss = running_style_loss / num_batches if num_batches else 0

        # --- Validation loop ---
        model_vae.eval()
        val_loss = 0.0
        run_val_recon_loss = 0.0
        run_val_kl_loss = 0.0
        run_val_content_loss = 0.0
        run_val_style_loss = 0.0

        val_batches = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                val_images = data["image"].to(args.device)

                recon_images, mean, logvar = model_vae(val_images)
                loss, val_recon_loss, val_content_loss, val_style_loss, val_kl_loss = vae_perceptual_loss(recon_images, val_images, mean, 
                                                                                                          logvar, perceptual_model, 
                                                                                                          content_layers=['relu_2', 'relu_3', 'relu_5'], 
                                                                                                          kl_beta=args.beta_kl_loss, style_weight=args.style_weight, content_weight=args.content_weight)

                run_val_recon_loss += val_recon_loss.item()
                run_val_kl_loss += val_kl_loss.item()
                val_loss += loss.item()
                run_val_content_loss += val_content_loss.item()
                run_val_style_loss += val_style_loss.item()

                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches else float('inf')
        avg_val_recon_loss = run_val_recon_loss / val_batches if val_batches else float('inf')
        avg_val_kl_loss = run_val_kl_loss / val_batches if val_batches else float('inf')
        avg_val_content_loss = run_val_content_loss / val_batches if val_batches else float('inf')
        avg_val_style_loss = run_val_style_loss / val_batches if val_batches else float('inf')

        print(f"Epoch {epoch}: Avg loss={avg_loss:.4f}, Avg Reconstruction={avg_recon_loss:.4f}, Avg KL={avg_kl_loss:.4f}, Avg content={avg_content_loss:.4f},")
        
        test_image = testset[0]['image']
        test_image = torch.tensor(test_image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

        with torch.no_grad():
            reconstructed_image, _, _ = model_vae(test_image)
            reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
            reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))  # Change to HWC format
            reconstructed_image = (reconstructed_image * 255)#.astype(np.uint8)  # Convert to uint8

        wandb.log({
            "Sample": wandb.Image(reconstructed_image), 
            "epoch_train_loss": avg_loss, "train_recon_loss": avg_recon_loss, "train_kl_loss": avg_kl_loss, "train_content_loss": avg_content_loss, "train_style_loss": avg_style_loss, 
            "epoch_val_loss": avg_val_loss, "val_recon_loss": avg_val_recon_loss, "val_kl_loss": avg_val_kl_loss, "val_content_loss": avg_val_content_loss, "val_style_loss": avg_val_style_loss, }, step=epoch)

        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            checkpoint_name = f"weights_ck_{epoch}.pt"
            torch.save(model_vae.state_dict(), os.path.join(path_to_save_checkpoints, checkpoint_name))
            print(f"Checkpoint saved: {checkpoint_name}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        # Early stopping
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break
    wandb.finish()
    return model_vae

if __name__ == '__main__':
    args = parse_args()
    train_vae(args)