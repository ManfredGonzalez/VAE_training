import torch.nn as nn
import torch
from encoder import VAE_Encoder
from decoder import VAE_Decoder

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # The encoder expects noise with shape (Batch_Size, 4, Height/8, Width/8).
        noise = torch.randn(batch_size, 4, height // 8, width // 8, device=x.device)
        latent, mean, logvar = self.encoder(x, noise)
        reconstruction = self.decoder(latent)
        return reconstruction, mean, logvar
