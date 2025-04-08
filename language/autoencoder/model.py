import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
#import pytorch_lightning as pl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from eval.colormaps import apply_pca_colormap
import matplotlib.pyplot as plt
import open_clip
import torchvision.models as models
from sklearn.decomposition import IncrementalPCA

class AutoencoderMLP(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims, clip_dim=768):
        super(AutoencoderMLP, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(clip_dim, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        
        #encoder_layers.append(nn.Sigmoid())
        self.encoder = nn.ModuleList(encoder_layers)
        
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        
        self.decoder = nn.ModuleList(decoder_layers)
       #print(self.encoder, self.decoder)
    
    def forward(self, x):
        lanet = self.encode(x)
        x = self.decode(lanet)
        # for m in self.encoder:
        #     x = m(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        # for m in self.decoder:
        #     x = m(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        return x

    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
class AutoencoderLight(pl.LightningModule):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims, in_channel_dim=768, is_MLP=True):
        super(AutoencoderLight, self).__init__()
        self.isMLP = is_MLP
        
        if self.isMLP:
            self.model = AutoencoderMLP(encoder_hidden_dims, decoder_hidden_dims, in_channel_dim)
        else:
            pass
        #self.l2_loss = nn.MSELoss()
        #self.cos_loss = nn.CosineEmbeddingLoss() #nn.CosineSimilarity(dim=1) #nn.CosineEmbeddingLoss()
    
    @torch.no_grad()
    def log_images(self, lang_gt, lang_recon, epoch, batch_idx, mode='train'):
        if mode == "train":
            global_step = epoch * len(self.trainer.train_dataloader) + batch_idx
        else:
            global_step = epoch * len(self.trainer.val_dataloaders) + batch_idx
        
        if len(lang_gt.shape) == 3:
            lang_gt = lang_gt.unsqueeze(0)
        
        if len(lang_recon.shape) == 3:
            lang_recon = lang_recon.unsqueeze(0)
            
        lang_gt_img = apply_pca_colormap(lang_gt[0].permute(1,2,0)).cpu()
        lang_recon_img = apply_pca_colormap(lang_recon[0].permute(1,2,0)).cpu()
        
        if global_step % 5 == 0:
            text_embs_chairs = self.get_user_embed(device="cpu", text="monitors")
            text_embs_floor = self.get_user_embed(device="cpu", text="floor")
            #print("lang recon shape: ", lang_recon.permute(0, 2, 3, 1).to("cpu").shape)
            sim_norm_orig = self.perform_similarity(lang_gt.permute(0, 2, 3, 1).to("cpu"), text_embs_chairs)
            sim_norm_recon = self.perform_similarity(lang_recon.permute(0, 2, 3, 1).to("cpu"), text_embs_chairs)
            
            
            #sim_norm_recon_floor = self.perform_similarity(lang_recon.permute(0, 2, 3, 1).to("cpu"), text_embs_floor)
            
            cmap = plt.get_cmap("turbo")
            heatmap = cmap(sim_norm_recon.detach().cpu().numpy())
            heatmap_orig = cmap(sim_norm_orig.detach().cpu().numpy())
            heatmap_rgb = heatmap[..., :3] 
            heatmap_orig_rgb = heatmap_orig[..., :3]
            
            self.logger.experiment.add_image(f'{mode}/_recon_table', heatmap_rgb.transpose(2, 0, 1), global_step)
            self.logger.experiment.add_image(f'{mode}/_orig_table', heatmap_orig_rgb.transpose(2, 0, 1), global_step)

        
        self.logger.experiment.add_images(f'{mode}/_lang_gt', lang_gt_img.unsqueeze(0).permute(0,3,1,2).detach().cpu().numpy(), global_step)
        self.logger.experiment.add_images(f'{mode}/_lang_recon', lang_recon_img.unsqueeze(0).permute(0,3,1,2).detach().cpu().numpy(), global_step)
    
    @torch.no_grad()
    def perform_similarity(self, clip_viz_dense, text_embs):
        sims = clip_viz_dense @ text_embs.T
        sims = sims.squeeze()
        sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
        return sim_norm

    @torch.no_grad()
    def get_user_embed(self, device="cuda", text="door"):
        name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
        tokenizer = open_clip.get_tokenizer(name)
        texts = tokenizer([text]).to(device)
        #print("texts shape: ", texts.shape)
        clip_model, _, _ = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device=device,)
        text_embs = clip_model.encode_text(texts)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        return text_embs

    def log_loss(self, l2loss, cosloss, loss, mode='train'):
        self.log(mode+'_loss/l2_loss', l2loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(mode+'_loss/cos_loss', cosloss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(mode+'_loss/total_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        #log lr
        if mode == 'train':
            self.log('lr', self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
    
    def forward(self, x):
        return self.model.forward(x)
    
    def encode(self, x):
        return self.model.encode(x)  

    def decode(self, x):
        return self.model.decode(x) 
    
    def get_low_dim(self, x):
        feat_extract = self.model.backbone(x)
        return self.model.encode(feat_extract)
    
    def get_high_dim(self, x):
        return self.model.decode(x)

    def l2_loss(self, network_output, gt):
        #gt = gt.view(-1, 768)
        return ((network_output - gt) ** 2).mean()

    def cos_loss(self, network_output, gt):
        #gt = gt.view(-1, 768)
        return 1 - F.cosine_similarity(network_output, gt, dim=1).mean()

    def on_train_start(self):
        pass
        #calculate the step size properly
        # num_cycles = 3
        # total_train_step = self.trainer.max_epochs * len(self.trainer.train_dataloader)
        # step_size = total_train_step // (num_cycles * 2)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer, 
        #     base_lr=1e-5, 
        #     max_lr=2e-2, 
        #     step_size_up=step_size, 
        #     mode='triangular2', 
        #     cycle_momentum=False
        # )
        # self.lr_scheduler = scheduler
        
    def random_reshape(self, data, target_height, target_width):
        """Randomly reshape input data, similar to how it may be reshaped before being input into the decoder."""
        _, feat_height, feat_width = data.shape

        # Random scaling factor to simulate reshaping during training
        scaling_factor = min(target_height / feat_height, target_width / feat_width)
        new_height = int(feat_height * scaling_factor)
        new_width = int(feat_width * scaling_factor)
        
        # Reshape the data using interpolation to simulate a resize
        reshaped_data = F.interpolate(data.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
        
        # Optionally, pad or crop to ensure consistent size
        if new_height < target_height or new_width < target_width:
            reshaped_data = F.pad(reshaped_data, (0, target_width - new_width, 0, target_height - new_height))
        elif new_height > target_height or new_width > target_width:
            reshaped_data = reshaped_data[:, :target_height, :target_width]

        return reshaped_data
    
    def training_step(self, batch, batch_idx):
        #reshape 
        N, C, H, W = batch.shape
        if self.isMLP:
            batch_reshape = batch.permute(0, 2, 3, 1)
            reshape_data = batch_reshape.reshape(-1, 768)
            outputs_dim3 = self.model.encode(reshape_data)
        else:
            reshape_data = batch
            #outputs_dim3 = self.model.encode(reshape_data)
        
        outputs = self.model(reshape_data)
        
        #outputs = F.interpolate(outputs, size=(192, 192), mode='bilinear', align_corners=False)
        
        l2loss = self.l2_loss(outputs, reshape_data)
        cosloss = self.cos_loss(outputs, reshape_data)
        total_loss = l2loss + 0.001*cosloss
        
        self.log_loss(l2loss, cosloss, total_loss, mode='train')
        
        if self.isMLP:
            outputs = outputs.view(N, H, W, C).permute(0, 3, 1, 2)
        else:
            outputs = outputs
        if batch_idx % 50 == 0:
            self.log_images(batch[0], outputs[0], self.current_epoch, batch_idx, mode='train')
        
        return total_loss 
    
    def validation_step(self, batch, batch_idx):
        N, C, H, W = batch.shape
        
        if self.isMLP:
            batch_reshape = batch.permute(0, 2, 3, 1)
            reshape_data = batch_reshape.reshape(-1, 768)
            outputs_dim3 = self.model.encode(reshape_data)
            outputs = self.model.decode(outputs_dim3)
        else:
            reshape_data = batch
            # outputs_dim3 = self.model.encode(reshape_data)
            # outputs = self.model.decode(outputs_dim3)
            outputs = self.model(reshape_data)
        
        #reshape_batch = batch.view(-1, 768)
        l2loss = self.l2_loss(outputs, reshape_data)
        cosloss = self.cos_loss(outputs, reshape_data)
        total_loss = l2loss + 0.001*cosloss
        
        self.log_loss(l2loss, cosloss, total_loss, mode='val')
        #visuliazation
        if self.isMLP:
            outputs = outputs.view(N, H, W, C).permute(0, 3, 1, 2)
        else:
            outputs = outputs
        self.log_images(batch[0], outputs[0], self.current_epoch, batch_idx, mode='val')
        
        return total_loss

    
    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=4e-4, weight_decay=1e-5)
        self.optimizer = optimizer
        
        warmup_iters = 50
        warmup_factor = 1.0
        
        def lr_lambda(current_step):
            if current_step < warmup_iters:
                return warmup_factor + (1.0 - warmup_factor) * (current_step / warmup_iters)
            return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6000, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iters])
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6000, eta_min=1e-6)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'step',
                'frequency': 1
            }
        }

        # optimizer = torch.optim.Adam(self.model.parameters(), 
        #                              lr=1e-4)
        # self.optimizer = optimizer
        # # Cyclic Learning Rate Scheduler
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     self.optimizer, 
        #     base_lr=1e-5, 
        #     max_lr=1e-2, 
        #     step_size_up=5, 
        #     mode='triangular2', 
        #     cycle_momentum=False
        # )
        
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss',
        #     }
        # }

class EncoderDecoderOnline(nn.Module):
    def __init__(self, method='mlp', input_dim=32, compressed_dim=15):
        super(EncoderDecoderOnline, self).__init__()
        self.method = method
        if method == 'mlp':
            # Define MLP-based encoder-decoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 24),
                nn.ReLU(),
                nn.Linear(24, 15),
                #nn.ReLU(),
                #nn.Linear(12, 6)
            )
            self.decoder = nn.Sequential(
                #nn.Linear(6, 12),
                #nn.ReLU(),
                nn.Linear(15, 24),
                nn.ReLU(),
                nn.Linear(24, input_dim)
            )
        elif method == 'pca': # We found mlp autoencoder is better than PCA
            # Placeholder for PCA model, trained incrementally
            self.pca = IncrementalPCA(n_components=compressed_dim)
            self.compressed_dim = compressed_dim
            self.is_fitted = False

    def encode(self, x):
        if self.method == 'mlp':
            x = self.encoder(x)
            return x / x.norm(dim=-1, keepdim=True)
        elif self.method == 'pca':
            if not self.is_fitted:
                raise ValueError("PCA model has not been fitted yet.")
            x_np = x.cpu().numpy()  # Convert to NumPy for PCA
            x_compressed = self.pca.transform(x_np)
            return torch.tensor(x_compressed, device=x.device).float()

    def decode(self, x):
        if self.method == 'mlp':
            x = self.decoder(x)
            return x / x.norm(dim=-1, keepdim=True)
        elif self.method == 'pca':
            if not self.is_fitted:
                raise ValueError("PCA model has not been fitted yet.")
            x_np = x.cpu().numpy()
            x_reconstructed = self.pca.inverse_transform(x_np)
            return torch.tensor(x_reconstructed, device=x.device).float()

    def incremental_fit(self, x):
        if self.method != 'pca':
            raise ValueError("Incremental fitting is only applicable for PCA.")
        x_np = x.cpu().numpy()
        self.pca.partial_fit(x_np)
        self.is_fitted = True
        