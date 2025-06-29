from encoder import VAE_Encoder
from decoder import VAE_Decoder
from vq_embedding import VQEmbedding

import torch.nn as nn
import torch.nn.functional as F

loss_fn = nn.MSELoss(reduction="sum")

def vqvae_loss(recon_x, x, vq_loss):
    #recon_loss = F.mse_loss(recon_x, x)

    recon_loss = loss_fn(recon_x, x)
    return recon_loss + vq_loss, recon_loss, vq_loss

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = VAE_Encoder()
        self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = VAE_Decoder()
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _, commitment_loss, codebook_loss = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, commitment_loss, codebook_loss