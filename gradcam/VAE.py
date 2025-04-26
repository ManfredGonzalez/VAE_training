import torch.nn as nn
import torch
from encoder import VAE_Encoder
from decoder import VAE_Decoder

def vae_loss(reconstructed, original, mean, logvar, kl_beta=0.1):
    """
    Summation-based BCE + KL divergence, matching the style from your image.

    reconstructed: [batch_size, channels, height, width]
    original: [batch_size, channels, height, width]
    mean, logvar: [batch_size, latent_dim]
    """

    b_size = reconstructed.size(0)

    #if loss == "bce":
    #    loss_fn = nn.BCELoss(reduction='sum')
        # Flatten for BCE
        
    #    reconstructed = reconstructed.view(b_size, -1)  # [B, D]
    #    original = original.view(b_size, -1)            # [B, D]

        # If your model outputs raw logits, you should use BCEWithLogitsLoss instead
        # or apply a final sigmoid here:
    #    reconstructed = torch.sigmoid(reconstructed)

        # 1) Reconstruction loss
    #    reconstruction_loss = loss_fn(reconstructed, original)
    #else:
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
    loss = reconstruction_loss + kl_beta * KL_DIV

    # Often we divide by batch size to get a mean loss per sample
    return loss / b_size, reconstruction_loss, KL_DIV

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # The encoder expects noise with shape (Batch_Size, 4, Height/8, Width/8).
        noise = torch.randn((batch_size, 4, height // 8, width // 8), device=x.device)
        latent, mean, logvar = self.encoder(x, noise)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, logvar
