import os
import argparse
import torch
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset import PineappleDataset  # Replace with your actual import
import optuna
from optuna import Trial
from vae.vq_vae import VQVAE, vqvae_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE on Pineapple Dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Checkpoint save path")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training ratio")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="Commitment cost weight") # Using optuna
    parser.add_argument("--device", type=str, default="cuda", help="Checkpoint save path")
    parser.add_argument("--optuna_trials_path_save", type=str, default="./optuna.csv", help="Optuna Path to save the trials")

    return parser.parse_args()

def setup_wandb(args, commitment_cost=0.1):
    """Login to Weights & Biases and initialize a new run."""
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(
        entity="imagine-laboratory-conare",
        project="vqvae_training",
        config={
            "learning_rate": args.lr,
            "architecture": "MLP",
            "dataset": "Pineapples",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "commitment_cost": commitment_cost
        },
    )
    return run


def train_vae(args, commitment_cost=0.1):

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoints, exist_ok=True)

    setup_wandb(args, commitment_cost=commitment_cost)

    # Load data
    trainset = PineappleDataset(train=True, train_ratio=args.train_ratio)
    testset = PineappleDataset(train=False, train_ratio=args.train_ratio)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_vae = VQVAE(commitment_cost=commitment_cost).to(args.device)
    optimizer = optim.Adam(model_vae.parameters(), lr=args.lr)

    global_step = 0
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):

        model_vae.train()

        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_vq_loss = 0.0
        running_commitment_loss = 0.0
        running_codebook_loss = 0.0

        num_batches = 0

        with tqdm(total=len(trainset), desc=f'Epoch: {epoch}/{args.epochs}', unit='img') as prog_bar:
            for i, data in enumerate(trainloader):
                color_images = data["image"].cuda()
                optimizer.zero_grad()

                reconstruction, vq_loss, commitment_loss, codebook_loss  = model_vae(color_images)
                loss, reconstruction_loss, vq_loss= vqvae_loss(reconstruction, color_images, vq_loss)

                loss.backward()
                optimizer.step()

                #wandb.log({"step_loss": loss.item()}, step=global_step)
                global_step += 1

                prog_bar.set_postfix(loss=loss.item())
                prog_bar.update(color_images.size(0))

                running_loss += loss.item()
                running_reconstruction_loss += reconstruction_loss.item()
                running_vq_loss += vq_loss.item()

                running_commitment_loss += commitment_loss.item()
                running_codebook_loss += codebook_loss.item()


                num_batches += 1
    
        avg_loss = running_loss / num_batches if num_batches else 0
        avg_recon_loss = running_reconstruction_loss / num_batches if num_batches else 0
        avg_vq_loss = running_vq_loss / num_batches if num_batches else 0
        avg_commitment_loss = running_commitment_loss / num_batches if num_batches else 0
        avg_codebook_loss = running_codebook_loss / num_batches if num_batches else 0

        # --- Validation loop ---
        model_vae.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                val_images = data["image"].to(args.device)

                reconstruction, vq_loss, commitment_loss, codebook_loss  = model_vae(val_images)
                loss, recon_loss, vq_loss= vqvae_loss(reconstruction, val_images, vq_loss)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches else float('inf')

        print(f"Epoch {epoch}: Train Average loss={avg_loss:.4f}, Average Reconstruction={avg_recon_loss:.4f}, Average VQ loss={avg_vq_loss:.4f}, Average Commitment loss={avg_commitment_loss:.4f}, Avg. Codebook loss={avg_codebook_loss:.4f}, Val Average loss={avg_val_loss:.4f},")
        wandb.log({
            "epoch_train_loss": avg_loss, "epoch_val_loss": avg_val_loss, "train_reconstruction_loss": avg_recon_loss, "train_vq_loss": avg_vq_loss, 
            "train_commitment_loss": avg_commitment_loss, "train_codebook_loss": avg_codebook_loss}, step=epoch)

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
    return model_vae

def test_vae(args, model_vae): #, last_checkpoint="./checkpoints/weights_ck_1.pt"):
    #model_vae = VQVAE()
    #checkpoint_path = last_checkpoint  # Replace with your checkpoint file
    #checkpoint = torch.load(checkpoint_path)
    #model_vae.load_state_dict(checkpoint)
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

            reconstruction, vq_loss = model_vae(val_images)
            loss, _, _ = vqvae_loss(reconstruction, val_images, vq_loss)

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
    commitment_cost = trial.suggest_float("commitment_cost", 0.1, 1.0, step=0.05)
    model_vae = train_vae(args, commitment_cost=commitment_cost)
    avg_val_loss = test_vae(args, model_vae) #, last_checkpoint="./checkpoints/weights_ck_1.pt")
    return avg_val_loss


def create_study():
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    df = study.trials_dataframe()
    df.to_csv("optuna_trials_vqvae.csv", index=False)

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