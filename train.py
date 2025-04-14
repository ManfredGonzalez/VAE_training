import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset import PineappleDataset  # Replace with your actual import
from VAE import VAE, vae_loss  # Replace with your actual import
import optuna
from optuna import Trial

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on Pineapple Dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Checkpoint save path")
    parser.add_argument("--train_ratio", type=float, default=0.1, help="Checkpoint save path")
    #parser.add_argument("--beta_kl_loss", type=float, default=0.1, help="Checkpoint save path") # Using optuna
    parser.add_argument("--device", type=str, default="cuda", help="Checkpoint save path")

    return parser.parse_args()

def setup_wandb(args, kl_beta=0.1):
    """Login to Weights & Biases and initialize a new run."""
    api_key = "0200cb5286fa9cfa203979ab3fdafc7ebee1fc8f" #os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(
        entity="dnnxl",
        project="VAE_Testing",
        config={
            "learning_rate": args.lr,
            "architecture": "MLP",
            "dataset": "Pineapples",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "beta_kl_loss": kl_beta
        },
    )
    return run


def train_vae(args, kl_beta=0.1):

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoints, exist_ok=True)

    setup_wandb(args)

    # Load data
    trainset = PineappleDataset(train=True, train_ratio=args.train_ratio)
    testset = PineappleDataset(train=False, train_ratio=args.train_ratio)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_vae = VAE().to(args.device)
    optimizer = optim.Adam(model_vae.parameters(), lr=args.lr)

    global_step = 0
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kl_loss = 0.0

        num_batches = 0

        with tqdm(total=len(trainset), desc=f'Epoch: {epoch}/{args.epochs}', unit='img') as prog_bar:
            for i, data in enumerate(trainloader):
                color_images = data["image"].cuda()
                optimizer.zero_grad()

                reconstruction, mean, logvar = model_vae(color_images)
                loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, color_images, mean, logvar, kl_beta=kl_beta)

                loss.backward()
                optimizer.step()

                #wandb.log({"step_loss": loss.item()}, step=global_step)
                global_step += 1

                prog_bar.set_postfix(loss=loss.item())
                prog_bar.update(color_images.size(0))

                running_loss += loss.item()
                running_reconstruction_loss += reconstruction_loss.item()
                running_kl_loss += kl_loss.item()

                num_batches += 1
    
        avg_loss = running_loss / num_batches if num_batches else 0
        avg_recon_loss = running_reconstruction_loss / num_batches if num_batches else 0
        avg_kl_loss = running_kl_loss / num_batches if num_batches else 0

        # --- Validation loop ---
        model_vae.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                val_images = data["image"].to(args.device)

                recon_images, mean, logvar = model_vae(val_images)
                loss, _, _ = vae_loss(recon_images, val_images, mean, logvar)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches else float('inf')

        print(f"Epoch {epoch}: Train Average loss={avg_loss:.4f}, Average Reconstruction={avg_recon_loss:.4f}, Average KL={avg_kl_loss:.4f}, Val Average loss={avg_val_loss:.4f},")
        wandb.log({"epoch_train_loss": avg_loss, "epoch_val_loss": avg_val_loss, "train_reconstruction_loss": avg_recon_loss, "train_kl_loss": avg_kl_loss}, step=epoch)

        # Checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            checkpoint_name = f"weights_ck_{epoch}.pt"
            torch.save(model_vae.state_dict(), os.path.join(args.checkpoints, checkpoint_name))
            print(f"Checkpoint saved: {checkpoint_name}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")

        # Early stopping
        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break
    wandb.finish()

def test_vae(args, last_checkpoint="./checkpoints/weights_ck_1.pt"):
    model_vae = VAE()
    checkpoint_path = last_checkpoint  # Replace with your checkpoint file
    checkpoint = torch.load(checkpoint_path)
    model_vae.load_state_dict(checkpoint)
    model_vae.eval()
    testset = PineappleDataset(train=False, train_ratio=args.train_ratio)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model_vae.eval()
    model_vae.to(args.device)
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            val_images = data["image"].to(args.device)

            recon_images, mean, logvar = model_vae(val_images)
            loss, _, _ = vae_loss(recon_images, val_images, mean, logvar)

            val_loss += loss.item()
            val_batches += 1

    avg_val_loss = val_loss / val_batches if val_batches else float('inf')
    return avg_val_loss



def objective(trial:Trial):
    """
    Objective function for Optuna to optimize SSIM by tuning loss weights.
    """
    global args
    # Suggested hyperparameters
    kl_beta = trial.suggest_float("kl_beta", 0.1, 1.0, step=0.1)
    train_vae(args, kl_beta=kl_beta)
    avg_val_loss = test_vae(args, last_checkpoint="./checkpoints/weights_ck_1.pt")
    return avg_val_loss


def create_study():
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    df = study.trials_dataframe()
    df.to_csv("optuna_trials.csv", index=False)

    print("✅ Best trial:")
    print(study.best_trial)

def main():
    train_vae(args)

if __name__ == '__main__':
    global args
    # Check if CUDA (GPU) is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    args = parse_args()
    #main()
    create_study()