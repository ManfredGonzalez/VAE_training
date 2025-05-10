import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper: Normalize for VGG
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

def normalize_for_vgg(x):
    return (x - imagenet_mean) / imagenet_std

# Helper: Gram Matrix for style loss
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G #/ (c * h * w)

# VGG feature extractor builder
def build_vgg_perceptual_model(layers=None):
    if layers is None:
        layers = ['relu_13', 'relu_14', 'relu_15', 'relu_16']
        
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    selected_layers = {}
    model = nn.Sequential().to(device)
    relu_count = 0
    conv_count = 0

    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            name = f'conv_{conv_count}'
        elif isinstance(layer, nn.ReLU):
            relu_count += 1
            name = f'relu_{relu_count}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{relu_count}'
        else:
            continue

        model.add_module(name, layer)
        if name in layers:
            selected_layers[name] = copy.deepcopy(model)
        if len(selected_layers) == len(layers):
            break

    return selected_layers

# Combined perceptual loss with content + style + KL
def vae_perceptual_loss(reconstructed, original, mean, logvar, perceptual_model, kl_beta=0.1,
                        content_layers=['relu_2', 'relu_3', 'relu_5'], style_layers=['relu_14', 'relu_15', 'relu_16'],
                        style_weight=0.0, content_weight=1.0):
    b_size = reconstructed.size(0)
    loss_fn = nn.MSELoss(reduction='mean')

    # Normalize inputs for VGG
    original = normalize_for_vgg(original.to(device))
    reconstructed = normalize_for_vgg(reconstructed.to(device))

    # Feature extraction
    feats_orig = {}
    feats_recon = {}
    for name, submodel in perceptual_model.items():
        with torch.no_grad():
            feats_orig[name] = submodel(original).detach()
        feats_recon[name] = submodel(reconstructed)

    # Content loss from multiple layers
    content_loss = 0.0
    for layer in content_layers:
        b, c, h, w = feats_recon[layer].size()
        content_loss += loss_fn(feats_recon[layer], feats_orig[layer]) #/ c #/ (c * h * w)
        
    # Style loss
    style_loss = 0.0
    for layer in style_layers:
        G1 = gram_matrix(feats_orig[layer])
        G2 = gram_matrix(feats_recon[layer])
        style_loss += loss_fn(G1, G2)

    # Pixel-space reconstruction loss
    pixel_recon_loss = loss_fn(reconstructed, original)

    # KL divergence
    kl_div = -torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    content_loss = content_weight * content_loss
    style_loss = style_weight * style_loss
    kl_div = kl_beta * kl_div
    # Weighted total loss
    total_loss = pixel_recon_loss + content_loss + style_loss + kl_div
    
    
    return total_loss / b_size, pixel_recon_loss / b_size, content_loss / b_size, style_loss / b_size, kl_div / b_size
