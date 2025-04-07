import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class AttentionFusion(nn.Module):
    def __init__(self, in_channels_high_res, in_channels_low_res):
        super(AttentionFusion, self).__init__()

        #self.align_channels = None
        if in_channels_high_res != in_channels_low_res:
            self.low_res_align = nn.Conv2d(in_channels_low_res, in_channels_high_res, kernel_size=1)
        else:
            self.low_res_align = nn.Identity()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_high_res * 2, in_channels_high_res, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels_high_res),
            nn.ReLU(inplace=True),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels_high_res, in_channels_high_res, kernel_size=3, padding=1), #3x3 conv
            nn.BatchNorm2d(in_channels_high_res),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_high_res, in_channels_high_res, kernel_size=1),
            nn.Sigmoid() #highlight important areas
        )

    def forward(self, high_res_feat, low_res_feat):
        #if self.align_channels:
        #low_res_feat = self.align_channels(low_res_feat)
        low_res_feat = self.low_res_align(low_res_feat)
        
        #concatenate feat
        fused_feat = torch.cat([high_res_feat, low_res_feat], dim=1)
        fused_feat = self.fusion(fused_feat)
        
        #apply attention
        attention_map = self.attention(fused_feat)
        out = fused_feat * attention_map + fused_feat #residual connection
        
        return out
    
class HighResLanguageFeatureNet(nn.Module):
    def __init__(self, desired_channels=768):
        super(HighResLanguageFeatureNet, self).__init__()
        
        # Initial projection to reduce channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        #upsample to 48x48
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.attention_fusion1 = AttentionFusion(in_channels_high_res=512, in_channels_low_res=384)
        
        #upsample to 96x96
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.attention_fusion2 = AttentionFusion(in_channels_high_res=256, in_channels_low_res=192)
        
        #upsample to 192x192
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(128, desired_channels, kernel_size=1)
    
    def forward(self, fv, f3, f2):
        x = self.initial_conv(fv)
        # Upsample fv from 24x24 to 48x48
        x = self.upsample1(x)

        # Resize f3 to match x's spatial size
        f3_resized = F.interpolate(f3, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Concatenate x and f3
        x = self.attention_fusion1(x, f3_resized)

        # Upsample to 96x96
        x = self.upsample2(x)

        # Resize f2 to match x's spatial size
        f2_resized = F.interpolate(f2, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        # Concatenate x and f2
        x = self.attention_fusion2(x, f2_resized)

        # Upsample to 192x192
        x = self.upsample3(x)

        # Final conv to get desired_channels
        x = self.final_conv(x)

        return x

class LangSupervisedNet(pl.LightningModule):
    def __init__(self,lambda_recon=1.0, lambda_edge=0.5, lambda_cosine=0.0, lambda_perceptual=0.0, lambda_tv=0.0):
        super().__init__()
        self.model = HighResLanguageFeatureNet()
        self.lambda_recon = lambda_recon
        self.lambda_edge = lambda_edge
        self.lambda_cosine = lambda_cosine
        self.lambda_tv = lambda_tv
        self.lambda_perpectual = lambda_perceptual
        
    def forward(self, fv, f3, f2):
        return self.model(fv, f3, f2)