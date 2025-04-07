from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from load_lang_model import load_lang_model
from autoencoder.model import Autoencoder, AutoencoderLangsplat, AutoencoderLight
from sed import add_sed_config
import numpy as np
import cv2
import argparse
import torch
import os
import open_clip
import tqdm
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from supervisedNet import LangSupervisedNet
import glob
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.colormaps import apply_pca_colormap
from sklearn.decomposition import PCA
import joblib
from skimage.metrics import structural_similarity as ssim
from hashlib import sha256
import torch.optim as optim
import torch.nn as nn
from sklearn.decomposition import IncrementalPCA

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_sed_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home/vision/Documents/GaussianGripMapping/language/configs/convnextL_768.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    return parser

def get_lang_feat(inputs, model):
    with torch.no_grad():
        _, dense_clip_viz = model(inputs)
        #clip_viz_dense = dense_clip_viz
    return dense_clip_viz

def get_user_embed(device="cuda", text="door"):
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
    print("text_embs shape: ", text_embs.shape)
    return text_embs

def save_npy_files(aug, model):
    input_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test/results"
    output_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test"

    ckpt_path = f'auto_ckpt/replica_room0/best_ckpt.pth'
    checkpoint = torch.load(ckpt_path)

    encoder_hidden_dims = [384, 192, 96, 48, 24, 12, 3]
    decoder_hidden_dims = [12, 24, 48, 96, 192, 384, 768]
    auto_model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    auto_model.load_state_dict(checkpoint)
    auto_model.eval()

    # Ensure the output directory exists
    
    lang_feat_dir = os.path.join(output_dir, "language_feat")
    lang_feat_dim3 = os.path.join(output_dir, "language_feat_dim3")
    os.makedirs(lang_feat_dir, exist_ok=True)
    os.makedirs(lang_feat_dim3, exist_ok=True)

    img_save_dir = os.path.join(lang_feat_dim3, "images")
    os.makedirs(img_save_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") and filename.startswith("frame"):
            image_path = os.path.join(input_dir, filename)
            original_image = read_image(image_path, format="BGR")
            height, width = original_image.shape[:2]
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to("cuda")
            inputs = [{"image": image, "height": height, "width": width}]
            clip_viz_dense = get_lang_feat(inputs, model)
            clip_viz_dense_dim3 = auto_model.encode(clip_viz_dense.permute(0,3,1,2))

            output_path_lang = os.path.join(lang_feat_dir, f"{os.path.splitext(filename)[0]}_f.npy")
            output_path_lang3 = os.path.join(lang_feat_dim3, f"{os.path.splitext(filename)[0]}_f.npy")

            clip_viz_dense_dim3_resize = F.interpolate(clip_viz_dense_dim3, size=(height, width), mode='bilinear', align_corners=False)
            #clip_viz_dense_dim3 = clip_viz_dense_dim3[0].cpu().detach().numpy()
            #scale up the image
            cv2.imwrite(os.path.join(img_save_dir, filename), clip_viz_dense_dim3_resize[0].permute(1,2,0).cpu().detach().numpy()*255)

            #np.save(output_path_lang, clip_viz_dense[0].cpu().numpy())
            #np.save(output_path_lang3, clip_viz_dense_dim3[0].cpu().detach().numpy())

def perform_similarity(clip_viz_dense, text_embs):
    sims = clip_viz_dense @ text_embs.T
    sims = sims.squeeze()
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    return sim_norm

@torch.no_grad()
def speed_inference_test(aug, model):
    input_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica/room0_test/results"

    times = []
    text_embs = get_user_embed()

    ckpt_path = f'auto_ckpt/replica_room0/best_ckpt.pth'
    checkpoint = torch.load(ckpt_path)

    encoder_hidden_dims = [384, 192, 96, 48, 24, 12, 3]
    decoder_hidden_dims = [12, 24, 48, 96, 192, 384, 768]
    auto_model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
    auto_model.load_state_dict(checkpoint)
    auto_model.eval()

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            original_image = read_image(image_path, format="BGR")
            height, width = original_image.shape[:2]
            image = aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to("cuda")
            inputs = [{"image": image, "height": height, "width": width}]
            t1 = time.time()
            clip_viz_dense = get_lang_feat(inputs, model)
            clip_viz_dense_dim3 = auto_model.encode(clip_viz_dense.permute(0,3,1,2))
            clip_viz_recon = auto_model.decode(clip_viz_dense_dim3)
            clip_viz_recon = clip_viz_recon.permute(0,2,3,1)
            elapsed_time = (time.time() - t1)*1000.0
            sim_norm_clip = perform_similarity(clip_viz_recon, text_embs)
            cmap = plt.get_cmap("turbo")
            heatmap_clip = cmap(sim_norm_clip.detach().cpu().numpy())
            plt.imsave("/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/rgb_heatmap/" + filename, heatmap_clip)
            print(f"Inference time for {filename}: {elapsed_time} ms")
            times.append(elapsed_time)


    print(f"Average inference time: {np.mean(times[1:])} ms")


@torch.no_grad()
def plot_lang_heatmap(img, lowres, highres, text, device, name="heatmap.png"):

    text_embs = get_user_embed(device=device, text=text)
    sim_norm_low_res = perform_similarity(lowres[0].permute(1,2,0), text_embs)
    sim_norm_high_res = perform_similarity(highres[0].permute(1,2,0), text_embs)

    cmap = plt.get_cmap("turbo")
    heatmap_low_res = cmap(sim_norm_low_res.detach().cpu().numpy())
    heatmap_high_res = cmap(sim_norm_high_res.detach().cpu().numpy())

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    ax[0].set_title("Image")
    ax[0].axis("off")

    ax[1].imshow(heatmap_low_res)
    ax[1].set_title("LowRes"+ text)
    ax[1].axis("off")

    ax[2].imshow(heatmap_high_res)
    ax[2].set_title("HighRes" + text)
    ax[2].axis("off")

    #plt.savefig(name)
    plt.show()

#save the high res features
def get_hr_feats(feat_sed, model_super, filename, output_dir, auto_model):
    img_name = os.path.basename(filename).split('.')[0]
    output_path = os.path.join(output_dir+"/hr_feat", f"{img_name}_hr.npy")
    high_res_lang = model_super(feat_sed['clip_vis_dense'], feat_sed['res3'], feat_sed['res2'])
    np.save(output_path, high_res_lang[0].detach().cpu().numpy())

    #auto encoder to reduce to 3 dims
    clip_viz_dense_dim3 = auto_model.encode(high_res_lang)
    #np.save(os.path.join(output_dir+"/hr_feat_dim3", f"{img_name}_hr.npy"), clip_viz_dense_dim3[0].detach().cpu().numpy())

class MLPEncoderDecoder(nn.Module):
    def __init__(self, input_dim=32, compressed_dim=12):
        super(MLPEncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, compressed_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x / x.norm(dim=-1, keepdim=True)  # Normalize
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        x = x / x.norm(dim=-1, keepdim=True)  # Normalize
        return x

    def forward(self, x):
        encoded = self.encode(x)
        return self.decode(encoded)


# Model Training and Evaluation
class ModelTrainer:
    def __init__(self, model, learning_rate=1e-2):
        self.model = model.to("cuda")
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, features, num_epochs=10):
        self.model.train()
        for _ in range(num_epochs):
            self.optimizer.zero_grad()
            # Forward pass
            reconstructed_features = self.model(features.detach())
            # Compute the loss
            l1loss = F.l1_loss(reconstructed_features, features)
            cosloss = 1 - F.cosine_similarity(reconstructed_features, features, dim=1).mean()
            loss = l1loss + 0.5 * cosloss
            #print(f"Loss: {loss.item()}")
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
        print(f"Final Loss: {loss.item()}")
        return loss.item()

class OnlineTestingPipeline:
    def __init__(self, model_trainer, auto_model, model_super, model_lang, aug):
        self.model_trainer = model_trainer
        self.auto_model = auto_model
        self.model_super = model_super
        self.model_lang = model_lang
        self.aug = aug
        self.buffer = []
        self.init_training = True
        self.pca = IncrementalPCA(n_components=12)

    def process_image(self, img_path):
        # Load and transform image
        original_image = read_image(img_path, format="BGR")
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        return torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to("cuda"), height, width

    def process_features(self, inputs):
        feat_sed = get_lang_feat(inputs, self.model_lang)
        high_res_lang = self.model_super(feat_sed['clip_vis_dense'], feat_sed['res3'], feat_sed['res2'])
        print("high_res_lang shape: ", high_res_lang.shape)
        batch_reshape = high_res_lang.permute(0, 2, 3, 1).view(-1, 768)
        return self.auto_model.encode(batch_reshape), feat_sed['clip_vis_dense']

    def incremental_train(self, features):
        epochs = 100 if self.init_training else 10
        features = features.detach()
        loss = self.model_trainer.train(features, num_epochs=epochs)
        self.init_training = False
        return loss

    def reconstruct_and_compare(self, compress_coco, fv_sed):
        with torch.no_grad():
            self.model_trainer.model.eval()
            compressed_feat = self.model_trainer.model.encode(compress_coco.to("cuda"))
            recon_compress = self.model_trainer.model.decode(compressed_feat)
            recon_train = self.auto_model.decode(recon_compress).view(1, 192, 192, 768)#.permute(0, 2, 3, 1)
            recon_coco = self.auto_model.decode(compress_coco).view(1, 192, 192, 768)#.permute(0, 2, 3, 1)
            fv_sed = fv_sed.permute(0,2,3,1)       
            self.visualize_similarity(recon_train, recon_coco, fv_sed)

    def visualize_similarity(self, recon_train, recon_coco, fv_sed):
        text_embs = get_user_embed(device="cuda", text="vase")
        sim_norm_recon = perform_similarity(recon_train, text_embs)
        sim_norm_recon_coco = perform_similarity(recon_coco, text_embs)
        sim_norm_sed = perform_similarity(fv_sed, text_embs)
    
        sim_norm_recon = (sim_norm_recon.detach().cpu().numpy() * 255).astype(np.uint8)
        sim_norm_recon_coco = (sim_norm_recon_coco.detach().cpu().numpy() * 255).astype(np.uint8)
        sim_norm_sed = (sim_norm_sed.detach().cpu().numpy() * 255).astype(np.uint8)

        # Apply TURBO colormap
        heatmap_recon = cv2.applyColorMap(sim_norm_recon, cv2.COLORMAP_TURBO)
        heatmap_recon_coco = cv2.applyColorMap(sim_norm_recon_coco, cv2.COLORMAP_TURBO)
        heatmap_sed = cv2.applyColorMap(sim_norm_sed, cv2.COLORMAP_TURBO)

        cv2.imshow("Reconstructed Similarity", heatmap_recon)
        cv2.imshow("Original Similarity", heatmap_sed)
        cv2.imshow("Reconstructed COCO Similarity", heatmap_recon_coco)
        cv2.waitKey(10)

        # cmap = plt.get_cmap("turbo")
        # heatmap = cmap(sim_norm_recon.detach().cpu().numpy())
        # heatmap_coco = cmap(sim_norm_recon_coco.detach().cpu().numpy())
        # heatmap_sed = cmap(sim_norm_sed.detach().cpu().numpy())

        # plt.subplot(1, 3, 1)
        # plt.title("Reconstructed Similarity")
        # plt.imshow(heatmap)

        # plt.subplot(1, 3, 2)
        # plt.title("Original Similarity")
        # plt.imshow(heatmap_sed)

        # plt.subplot(1, 3, 3)
        # plt.title("Reconstructed Coco Similarity")
        # plt.imshow(heatmap_coco)
        # plt.show()

    def run_pipeline(self, img_paths):
        for img_path in img_paths:
            # Load and process image
            image, height, width = self.process_image(img_path)
            inputs = [{"image": image, "height": height, "width": width}]
            compress_coco, fv_sed = self.process_features(inputs)
            self.buffer.append(compress_coco.cpu())

            # Incremental training when buffer is full
            if len(self.buffer) == 3:
                features = torch.cat(self.buffer, dim=0).to("cuda")
                loss = self.incremental_train(features)
                print(f"Training Loss: {loss}")
                self.buffer.pop(0)  # Keep buffer size constant

            # Reconstruct and visualize similarity after training
            self.reconstruct_and_compare(compress_coco, fv_sed)

@torch.no_grad()
def test_mlp_ae(auto_model, model_lang, aug, img_paths, mode="viz", user_queries=["vase"]):
    reconstruction_errors = {"MSE": [], "MAE": [], "Cosine Similarity": []}

    for img_path in img_paths:
        original_image = read_image(img_path, format="BGR")
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to("cuda")

        inputs = [{"image": image, "height": height, "width": width}]
        feat_sed = get_lang_feat(inputs, model_lang)
        high_res_lang = model_super(feat_sed['clip_vis_dense'], 
                                    feat_sed['res3'], feat_sed['res2'])
        fv_sed = feat_sed['clip_vis_dense']
        print("fv_sed shape: ", fv_sed.shape)
        # high_res_lang = np.load("/home/saimouli/Desktop/test/clip_viz_dense.npy")
        # high_res_lang = torch.from_numpy(high_res_lang).float().to("cuda")
        # print("high_res_lang shape: ", high_res_lang.shape)
        N, C, H, W = high_res_lang.shape
        batch_reshape = high_res_lang.permute(0, 2, 3, 1)
        high_res_lang_res = batch_reshape.view(-1, 768)
        #resize to 192x192
        fv_sed = F.interpolate(fv_sed, size=(192, 192), mode='bilinear', align_corners=False)
        fv_reshape = fv_sed.permute(0, 2, 3, 1).view(-1, 768)
        #compress = auto_model.encode(high_res_lang_res)
        compress = auto_model.encode(fv_reshape)

        #clip_viz_dense_dim3 = compress.T.view(15, H, W)
        #target_min = 0; target_max = 1; target_mean = 0.651042640209198
        #latent_code_adjusted = adjust_latent_code(clip_viz_dense_dim3, target_min, target_max, target_mean)
        #latent_code_original = undo_adjustment(latent_code_adjusted, target_min, target_max, target_mean)

        #clip_viz_dense_dim3_clipped = torch.sigmoid(clip_viz_dense_dim3)
        # gt_lang_feat_resize = F.interpolate(clip_viz_dense_dim3.unsqueeze(0), 
        #                                 size=(height, width), mode='bilinear', 
        #                                 align_corners=False).squeeze(0)
        #print("gt_lang_feat_resize shape: ", gt_lang_feat_resize.shape)
        # # #inverse sigmoid
        #epsilon = 1e-6
        #test = np.load("/home/saimouli/Desktop/test/gt_lang.npy")
        #test = torch.from_numpy(test).float().to("cuda")
        #print('test shape: ', test.shape)
        #latent_code_original = test
        #latent_code_original = torch.log((test + epsilon)/(1-test + epsilon))
        #print("latent_code_original shape: ", latent_code_original.shape)
        
        #latent_code_original_shape = gt_lang_feat_resize.permute(1, 2, 0).reshape(-1, 15)  # 
        recon = auto_model.decode(compress)
        #recon = recon.view(1, height, width, 768).permute(0, 3, 1, 2)
        recon = recon.view(N, H, W, C).permute(0, 3, 1, 2)


        # clip_viz_dense_dim3 = compress.T.view(15, H, W)
        # #gt_lang_feat = clip_viz_dense_dim3.cpu().detach()
        # gt_lang_feat = np.load("/home/saimouli/Desktop/test/gt_lang_feat.npy")
        # gt_lang_feat = torch.from_numpy(gt_lang_feat).float()
        # feat = gt_lang_feat.permute(1,2,0).to("cuda")
        # recon = auto_model.decode(feat.view(-1,15))
        # recon = recon.view(1,192,192,768)

        # text_embs = get_user_embed(device="cuda", text="door")
        # sim_norm_recon = perform_similarity(recon, text_embs)
        # cmap = plt.get_cmap("turbo")
        # heatmap = cmap(sim_norm_recon.detach().cpu().numpy())
        # heatmap_rgb = heatmap[:, :, :3]
        # heatmap_bgr = (heatmap_rgb * 255).astype(np.uint8)
        # heatmap_bgr = cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2BGR)
        # cv2.imshow("heatmap_clip", heatmap_bgr)
        # cv2.waitKey(0)
        

        #recon = auto_model.decode(compress)
        #recon = recon.T.view(N, C, H, W)

        #load compres feat only
        # compress = torch.from_numpy(np.load("results/datasets_Replica/2024-10-20-15-35-20/psnr/before_opt/lang/190.npy")).to("cuda")
        # compress = compress.permute(0,2,3,1).view(-1, 15)
        # recon = auto_model.decode(compress)
        # recon = recon.view(N, H, W, C).permute(0, 3, 1, 2)


        # load compress feat and reshape to gauss features
        #compress = torch.from_numpy(np.load("/media/saimouli/Data6T/Replica/room0/compress_feat_15_mlp_room1_2_train/frame000161_ld.npy")).to("cuda")
        #compress = F.interpolate(compress.unsqueeze(0), size=(680, 1200), mode='bilinear', align_corners=False)
        #compress = compress.permute(0,2,3,1)
        #print("Compress shape: ", compress.shape)
        # viz_compress_res = apply_pca_colormap(compress.permute(0,2,3,1))
        # plt.imshow(viz_compress_res[0].cpu().numpy())
        # plt.show()
        #compress = compress.view(-1, 15)
        #recon = auto_model.decode(compress)
        #recon = recon.view(1, 680, 1200, 768).permute(0, 3, 1, 2)
        #recon = recon.view(N, H, W, C).permute(0, 3, 1, 2)

        
        # viz_high_res = apply_pca_colormap(high_res_lang.permute(0,2,3,1))
        # viz_recon = apply_pca_colormap(recon.permute(0,2,3,1))
        # plt.subplot(1, 2, 1)
        # plt.imshow(viz_high_res[0].cpu().numpy())
        # plt.subplot(1, 2, 2)
        # plt.imshow(viz_recon[0].cpu().numpy())
        # plt.show()
        
        if mode == "eval":
            mse = torch.nn.functional.mse_loss(recon, high_res_lang).item()
            mae = F.l1_loss(recon, high_res_lang).item()
            cosine_similarity = F.cosine_similarity(recon.view(768,-1), high_res_lang.view(768,-1), dim=0).mean().item()
            reconstruction_errors["MSE"].append(mse)
            reconstruction_errors["MAE"].append(mae)
            reconstruction_errors["Cosine Similarity"].append(cosine_similarity)

            print(f"Metrics for {img_path}: MSE={mse}, MAE={mae}, Cosine Similarity={cosine_similarity}")
        
        elif mode == "viz":
            for user_query in user_queries:
                text_embs = get_user_embed(device="cuda", text=user_query)
                sim_norm_recon = perform_similarity(recon.permute(0,2,3,1), text_embs)
                sim_norm_orig = perform_similarity(high_res_lang.permute(0,2,3,1), text_embs)
                sim_norm_sed = perform_similarity(fv_sed.permute(0,2,3,1), text_embs)
                
                cmap = plt.get_cmap("turbo")
                heatmap = cmap(sim_norm_recon.detach().cpu().numpy())
                heatmap_orig = cmap(sim_norm_orig.detach().cpu().numpy())
                heatmap_sed = cmap(sim_norm_sed.detach().cpu().numpy())
                plt.subplot(1, 3, 1)
                plt.title("Reconstructed_"+str(user_query))
                plt.imshow(heatmap)
                plt.subplot(1, 3, 2)
                plt.title("Original high res")
                plt.imshow(heatmap_orig)
                plt.subplot(1, 3, 3)
                plt.title("Original low res")
                plt.imshow(heatmap_sed)
                plt.show()

                # heatmap_rgb = heatmap[:, :, :3]
                # heatmap_bgr = (heatmap_rgb * 255).astype(np.uint8)
                # heatmap_bgr = cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2BGR)
                # cv2.imshow("heatmap_clip", heatmap_bgr)
                # cv2.waitKey(0)
    
        if mode == "eval" and len(reconstruction_errors) > 0:
            avg_metrics = {metric: np.mean(values) for metric, values in reconstruction_errors.items()}
            print("\nAverage Metrics:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value}")

if __name__ == "__main__":

    check_pt_highres = "/home/saimouli/Desktop/Bosch/LangRes/tensorboard_logs/supervised/epoch=177-step=39541.ckpt"
    sed_lang_model_path = "/home/saimouli/Desktop/Bosch/LangRes/seg_clip_model_l.pth"

    img_folder = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/results" #room0_test/results
    output_dir = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/language_features"
    auto_ckpt_path = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/auto_encoder/auto_run1/epoch=133-step=2948.ckpt"
    #"/media/saimouli/RPNG_FLASH_4/datasets/Replica/office4_test/ae_encoder/epoch=145-step=3212.ckpt"
    #auto_ckpt_path = "/home/saimouli/Downloads/room0_small/auto_encoder/room0_langsplat_slam_convnext_large_d_320/best_ckpt.pth" 
    auto_ckpt_path = "/home/saimouli/Desktop/Bosch/GaussianGripMapping/autoencoder/version_5_code32/epoch=127-step=17536.ckpt"
    #auto_ckpt_path = "/home/saimouli/Desktop/Bosch/GaussianGripMapping/autoencoder/version_4_code64/checkpoints/epoch=149-step=20550.ckpt"
    #"/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/auto_encoder/langsplat_mlp_code15_room1_2_train/epoch=123-step=992.ckpt"
    #"/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/langslam/auto_encoder/code12/epoch=106-step=15836.ckpt"
    #"/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/auto_encoder/mlp_code15_overfit_room0/epoch=599-step=1800.ckpt"
    #"/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/auto_encoder/mlp_code15_room1_2_train/epoch=417-step=2090.ckpt"
    #auto_ckpt_path = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room1_test/ae_encoder/epoch=244-step=4410.ckpt"
    #"/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/langslam/auto_encoder/hr_auto/best_ckpt.pth"
    device = "cuda"

    ##high res autoencoder
    ##128 encoderq
    # encoder_hidden_dims = [512, 256, 128]
    # decoder_hidden_dims = [192, 256, 384, 512, 768]

    # #64 encoder
    # encoder_hidden_dims = [512, 256, 128, 64]
    # decoder_hidden_dims = [192, 256, 384, 512, 768]

    # encoder_hidden_dims = [384, 192, 96, 48, 24, 15]
    # decoder_hidden_dims = [24, 48, 96, 192, 384, 384, 768]

    encoder_hidden_dims = [512, 256, 128, 64, 32]
    decoder_hidden_dims = [192, 256, 384, 512, 768]
    
    in_channel_dim = 768
    auto_model = AutoencoderLight(encoder_hidden_dims, decoder_hidden_dims, in_channel_dim, is_MLP=True).to("cuda")
    auto_model = auto_model.load_from_checkpoint(auto_ckpt_path, 
                                  encoder_hidden_dims=encoder_hidden_dims, 
                                  decoder_hidden_dims=decoder_hidden_dims,
                                  in_channel_dim=in_channel_dim,
                                  is_MLP=True)
    
    #load_state_dict(torch.load(auto_ckpt_path))
    auto_model.to(device)
    auto_model.eval()
    # auto_model = AutoencoderLight(encoder_hidden_dims, decoder_hidden_dims, in_channel_dim)
    # ckpt = torch.load(auto_ckpt_path)
    # auto_model = auto_model.load_state_dict(ckpt)
    # auto_model.to(device)
    # auto_model.eval()

    model_super = LangSupervisedNet()
    model_super = model_super.load_from_checkpoint(check_pt_highres)
    model_super.to(device)
    model_super.eval()

    model_lang = load_lang_model(model_path=sed_lang_model_path)
    model_lang.to(device)
    model_lang.eval()

    aug = T.ResizeShortestEdge(
        [640, 640], 2560
    )

    img_paths = sorted(glob.glob(os.path.join(img_folder, 'frame*.jpg')))
    img_paths = img_paths[::5]
    
    model = MLPEncoderDecoder(input_dim=32, compressed_dim=12).to("cuda")
    trainer = ModelTrainer(model)

    pipeline = OnlineTestingPipeline(trainer, auto_model, model_super, model_lang, aug)
    pipeline.run_pipeline(img_paths)

    #get the list of image from the folder
    #os.makedirs(output_dir, exist_ok=True)
    #list of files in img_folder ending with .jpg
    #img_paths = glob.glob(os.path.join(img_folder, 'frame*.jpg'))

    # test_pca_feat(model_lang, model_super, 
    #               aug, "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/results")

    ##office0: ["blinds", "floor", "rug", "tabe", "tv-screen", "wall"]
    ##room2: ["blinds", "floor", "rug", "tabe", "tv-screen", "wall"]
    ##room0: ["blinds", "floor", "rug", "tabe", "tv-screen", "wall"]
    ##room1: ["blinds", "floor", "rug", "tabe", "tv-screen", "wall"]
    
    #img_folder = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/office4/results"
    # img_paths = sorted(glob.glob(os.path.join(img_folder, 'frame*.jpg')))
    # img_paths = img_paths[::10]
    # test_mlp_ae(auto_model, model_lang, aug, 
    #            img_paths, 
    #            mode="viz",
    #            user_queries=['wall', 'chair', 'floor', 'plate', 'vase', 'window', 'indoor-plant', 'table', 'blinds'])

    #gaussian_lang_dir = "results/datasets_Replica/2024-10-30-21-51-05/psnr/before_opt/lang"
     # # ##gaussian_langsplat_dir = "/home/saimouli/Downloads/test" #"/home/saimouli/Desktop/Bosch/GaussianGripMapping/results/datasets_Replica/load_highres/psnr/before_opt/lang"
     # # #pca_model = joblib.load("/home/saimouli/Desktop/Bosch/GaussianGripMapping/language/autoencoder/pca/pca_model_24.pkl")
    #gaussian_feats(gaussian_lang_dir, img_folder, auto_model, None, code_size=15)

    #hr_feat_dir = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/language_features/hr_feat"
    #test_hr_feats(hr_feat_dir, auto_model)


    #img_paths = glob.glob(os.path.join(img_folder, '*.png'))
    #img_paths = img_paths[20:25]

    # for filename in img_paths:
    #     #image_path = os.path.join(input_dir, filename)
    #     original_image = read_image(filename, format="BGR")
    #     height, width = original_image.shape[:2]
    #     image = aug.get_transform(original_image).apply_image(original_image)
    #     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    #     image.to("cuda")
    #     inputs = [{"image": image, "height": height, "width": width}]
    #     feat_sed = get_lang_feat(inputs, model_lang)

    #     get_hr_feats(feat_sed, model_super, filename, output_dir, auto_model)


        # high_res_lang = model_super(feat_sed['clip_vis_dense'], feat_sed['res3'], feat_sed['res2'])

        # ##inference
        # plot_features(image, feat_sed['clip_vis_dense'], high_res_lang, name="features.png")
        # plot_lang_heatmap(image, feat_sed['clip_vis_dense'], high_res_lang, "chairs", device=device, name="heatmap.png")




        #img_name = os.path.basename(filename).split('.')[0]
        #output_path = os.path.join(output_dir, f"{img_name}_f.npy")
        #np.save(output_path, high_res_lang[0].detach().cpu().numpy())
        #print(f"Saved {output_path}")

    #test_gaussian_lang_feat()

    #args = get_parser().parse_args()
    #cfg = setup_cfg(args)

    # t1 = time.time()
    # model = build_model(cfg)
    # model.eval()
    # checkpointer = DetectionCheckpointer(model)
    # checkpointer.load(cfg.MODEL.WEIGHTS)
    # print(f"Model building time: {(time.time() - t1)*1000.0} ms")

    #model_path="/home/vision/Documents/GaussianGripMapping/seg_clip_model_l.pth"
    #t1 = time.time()
    #model = load_lang_model(model_path=model_path)
    #print(f"Model loading time: {(time.time() - t1)} s")
    #print("Model created")

    # aug = T.ResizeShortestEdge(
    #     [640, 640], 2560
    # )
    #model.to("cuda")
    #test_model_speed(model, aug)
    #onnx_speed_test(aug)
    #optimize_onnx()

    # #input_format = cfg.INPUT.FORMAT
    # model.to("cuda")
    # save_npy_files(aug, model)
    #speed_inference_test(aug, model)

    # input_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/rgb"
    # output_dir = "/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/language_feat"

    # # Ensure the output directory exists
    # os.makedirs(output_dir, exist_ok=True)

    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".jpg") or filename.endswith(".png"):
    #         image_path = os.path.join(input_dir, filename)
    #         original_image = read_image(image_path, format="BGR")
    #         height, width = original_image.shape[:2]
    #         image = aug.get_transform(original_image).apply_image(original_image)
    #         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    #         image.to("cuda")
    #         inputs = [{"image": image, "height": height, "width": width}]
    #         clip_viz_dense = get_lang_feat(inputs, model)
    #         output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_f.npy")
    #         np.save(output_path, clip_viz_dense[0].cpu().numpy())

    # with torch.no_grad():
    #     original_image = read_image("/media/vision/RPNG_FLASH_4/datasets/Replica2/vmap/room_0/imap/00/rgb/rgb_141.png", format="BGR")
    #     if input_format == "RGB":
    #         original_image = original_image[:, :, ::-1]
        
    #     height, width = original_image.shape[:2]
    #     image = aug.get_transform(original_image).apply_image(original_image)
    #     image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    #     image.to("cuda")
    #     inputs = [{"image": image, "height": height, "width": width}]
    #     _, dense_clip_viz = model(inputs)
    #     print("dense_clip_viz shape: ", dense_clip_viz.shape)
    #     clip_viz_dense = dense_clip_viz.permute(0,2,3,1)
    #     name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    #     tokenizer = open_clip.get_tokenizer(name)
    #     texts = tokenizer(["window blinds"]).cuda()
    #     print("texts shape: ", texts.shape)
    #     clip_model, _, _ = open_clip.create_model_and_transforms(
    #             name,
    #             pretrained=pretrain,
    #             device="cuda",)
    #     text_embs = clip_model.encode_text(texts)
    #     text_embs /= text_embs.norm(dim=-1, keepdim=True)
    #     print("text_embs shape: ", text_embs.shape)

    #     sims = clip_viz_dense @ text_embs.T
    #     sims = sims.squeeze()
    #     sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    #     import matplotlib.pyplot as plt
    #     cmap = plt.get_cmap("turbo")
    #     heatmap = cmap(sim_norm.cpu().numpy())
    #     plt.imshow(heatmap)
    #     #plt.savefig("/home/vision/Documents/test_clip/heatmap.png")
    #     plt.show()

