import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.loss = nn.L1Loss()
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, pred, target):
        pred = pred.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        pred = self.preprocess(pred)
        target = self.preprocess(target)
        pred_vgg = self.vgg(pred)
        target_vgg = self.vgg(target)
        return self.loss(pred_vgg, target_vgg)

class CycleGANLoss(nn.Module):
    def __init__(self, device, lambda_cycle=10.0, lambda_identity=5.0, alpha=100, beta=1.0, gamma=10.0, delta=1.0, lambda_gan=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.vgg = VGGLoss(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.bce = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for stability with autocast
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.lambda_gan = lambda_gan

    def forward_generator(self, pred, target, d_fake):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        recon_loss = self.l1(pred, target)
        perceptual_loss = self.vgg(pred, target)
        ssim_loss = 1.0 - self.ssim(pred, target)
        mse_loss = self.mse(pred, target)
        
        # Adversarial loss: Encourage generator to fool discriminator
        gan_loss = self.bce(d_fake, torch.ones_like(d_fake))  # d_fake contains logits
        
        total_loss = (
            recon_loss +
            self.alpha * perceptual_loss +
            self.gamma * ssim_loss +
            self.delta * mse_loss +
            self.lambda_gan * gan_loss
        )
        
        if torch.isnan(total_loss):
            print("⚠️ Warning: NaN detected in generator loss. Returning zero loss.")
            total_loss = torch.tensor(0.0, requires_grad=True).to(pred.device)
        
        return total_loss

    def forward_discriminator(self, d_real, d_fake):
        # Discriminator loss with label smoothing for stability
        real_loss = self.bce(d_real, torch.full_like(d_real, 0.9))  # Smooth labels for real
        fake_loss = self.bce(d_fake, torch.zeros_like(d_fake))
        total_loss = (real_loss + fake_loss) / 2
        
        if torch.isnan(total_loss):
            print("⚠️ Warning: NaN detected in discriminator loss. Returning zero loss.")
            total_loss = torch.tensor(0.0, requires_grad=True).to(d_real.device)
        
        return total_loss