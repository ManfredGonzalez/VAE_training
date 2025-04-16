from encoder import VAE_Encoder
from decoder import VAE_Decoder
from vq_embedding import VQEmbedding

import torch.nn as nn

loss_fn = nn.MSELoss(reduction='sum')

def vqvae_loss(recon_x, x, vq_loss):
    
    recon_loss = loss_fn(recon_x, x)
    return recon_loss + vq_loss

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128):
        super(VQVAE, self).__init__()
        self.encoder = VAE_Encoder()
        self.vq_layer = VQEmbedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder = VAE_Decoder()
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, _ = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss