import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from autoencoder.model import AutoencoderLight
from load_lang_model import load_lang_model, get_lang_feat

import detectron2.data.transforms as T
import open_clip
import matplotlib.pyplot as plt
import time
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.colormaps import apply_pca_colormap
import glob
from supervisedNet import LangSupervisedNet

def get_user_embed():
    name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    tokenizer = open_clip.get_tokenizer(name)
    texts = tokenizer(["vase"]).cuda()
    #print("texts shape: ", texts.shape)
    clip_model, _, _ = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device="cuda",)
    text_embs = clip_model.encode_text(texts)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)
    print("text_embs shape: ", text_embs.shape)
    return text_embs

def perform_similarity(clip_viz_dense, text_embs):
    sims = clip_viz_dense @ text_embs.T
    sims = sims.squeeze()
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    return sim_norm

def resize_shortest_edge(tensor, min_size, max_size):
    _, h, w = tensor.shape
    scale = min(min_size / min(h, w), max_size / max(h, w))
    newh, neww = int(h * scale), int(w * scale)
    return F.resize(tensor, [newh, neww])

@torch.no_grad()
def get_f2_feats(image_path, auto_model, lang_model, aug):
    #encode(192,192,192)
    #decode(3,192,192)
    original_image = np.array(Image.open(image_path))
    height, width = original_image.shape[:2]
    print("height, width: ", height, width)

    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
    print("image shape: ", image.shape)

    inputs = [{"image": image, "height": height, "width": width}]
    clip_viz_dense = get_lang_feat(inputs, lang_model, is_lang=False)
    f2_feat = clip_viz_dense['res2']
    f2_feat_resize = F.interpolate(f2_feat, size=(height, width), mode='bilinear', align_corners=False)
    f2_feat_resize = apply_pca_colormap(f2_feat_resize.squeeze(0).permute(1,2,0).cpu())

    f2_feat_dim3 = auto_model.encode(f2_feat)
    f2_feat_dim3_resize = F.interpolate(f2_feat_dim3, size=(height, width), mode='bilinear', align_corners=False)
    f2_feat_dim3_resize = f2_feat_dim3_resize[0].permute(1,2,0).cpu().numpy()
    f2_feat_dim3_resize = (f2_feat_dim3_resize * 255).astype(np.uint8)
    f2_feat_dim3_resize = np.ascontiguousarray(f2_feat_dim3_resize)

    plt.subplot(1,2,1)
    plt.title("PCA orignal f2 feat")
    plt.imshow(f2_feat_resize.numpy())
    plt.subplot(1,2,2)
    plt.title("Recon F2")
    plt.imshow(f2_feat_dim3_resize)
    plt.show()

#either lang or f2 feats
@torch.no_grad()
def get_lang_feats(image_path, auto_model, lang_model, aug):
    original_image = np.array(Image.open(image_path))
    height, width = original_image.shape[:2]
    print("height, width: ", height, width)

    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
    print("image shape: ", image.shape)

    inputs = [{"image": image, "height": height, "width": width}]
    clip_viz_dense = get_lang_feat(inputs, lang_model, is_lang=True)
    clip_viz_dense_resize = F.interpolate(clip_viz_dense, size=(height, width), mode='bilinear', align_corners=False)
    clip_viz_dense_resize = clip_viz_dense_resize.permute(0, 2, 3, 1)

    clip_viz_dense_dim3 = auto_model.encode(clip_viz_dense)
    clip_viz_dense_dim3_resize = F.interpolate(clip_viz_dense_dim3, size=(height, width), mode='bilinear', align_corners=False)
    clip_viz = clip_viz_dense_dim3_resize[0].permute(1,2,0).cpu().numpy()
    clip_viz = (clip_viz * 255).astype(np.uint8)
    clip_viz = np.ascontiguousarray(clip_viz)

    clip_viz_recon = auto_model.decode(clip_viz_dense_dim3)
    clip_viz_recon_resize = F.interpolate(clip_viz_recon, size=(height, width), mode='bilinear', align_corners=False)
    clip_viz_recon_resize = clip_viz_recon_resize#.permute(0, 2, 3, 1)

    text_embs = get_user_embed()
    sim_norm_autoencoder = perform_similarity(clip_viz_recon_resize.permute(0,2,3,1), text_embs)
    sim_norm_clip = perform_similarity(clip_viz_dense_resize, text_embs)

    cmap = plt.get_cmap("turbo")
    heatmap_auto = cmap(sim_norm_autoencoder.cpu().numpy())
    heatmap_clip = cmap(sim_norm_clip.cpu().numpy())

    plt.subplot(1,4,1)
    plt.title("Clip heatmap")
    plt.imshow(heatmap_clip)
    plt.subplot(1,4,2)
    plt.title("Encoder heatmap")
    plt.imshow(heatmap_auto)
    plt.subplot(1,4,3)
    plt.title("Dim3 image")
    plt.imshow(clip_viz_dense_dim3[0].permute(1,2,0).cpu().numpy())
    plt.subplot(1,4,4)
    plt.title("Orig. image")
    plt.imshow(original_image)
    plt.show()

@torch.no_grad()
def plot_features(img, lowres, highres, name="features.png"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    ax[0].set_title("Image")
    ax[0].axis("off")

    lowres_pca = apply_pca_colormap(lowres.permute(0,2,3,1)).detach().cpu()
    ax[1].imshow(lowres_pca[0])
    ax[1].set_title("LowRes")
    ax[1].axis("off")
    
    highres_pca = apply_pca_colormap(highres.permute(0,2,3,1)).detach().cpu()
    ax[2].imshow(highres_pca[0])
    ax[2].set_title("HighRes")
    ax[2].axis("off")
    #plt.savefig(name)
    plt.show()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="replica_room0") #required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[384, 192, 96, 48, 24, 15, 3] #[192, 96, 48, 24, 12, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[24, 48, 96, 192, 384, 384, 768] #[12, 24, 48, 96, 192],
                    )
    parser.add_argument('--model_path', type=str, 
                        default="seg_clip_model_l.pth")
    parser.add_argument('--auto_ckpt', type=str, 
                        default="autoencoder/viz_code3_room0/epoch=49-step=100.ckpt")
    parser.add_argument('--in_channel_dim', type=int, default=768)
    parser.add_argument('--check_pt_highres', type=str, default="supervised/epoch=177-step=39541.ckpt")
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    checkpoint = args.auto_ckpt

    auto_model = AutoencoderLight(encoder_hidden_dims, decoder_hidden_dims, args.in_channel_dim, is_MLP=True).to("cuda:0")
    auto_model = auto_model.load_from_checkpoint(checkpoint, 
                                                 encoder_hidden_dims=encoder_hidden_dims, 
                                                 decoder_hidden_dims=decoder_hidden_dims, 
                                                 in_channel_dim=args.in_channel_dim,
                                                 is_MLP=True)
    auto_model.eval()
    print("Autoencoder model loaded successfully.")

    model_super = LangSupervisedNet()
    model_super = model_super.load_from_checkpoint(args.check_pt_highres)
    model_super.to("cuda:0")
    model_super.eval()

    t1 = time.time()
    lang_model = load_lang_model(model_path=args.model_path)
    print("Time taken to load lang model: ", time.time()-t1)

    aug = T.ResizeShortestEdge(
        [640,640], 2560
    )
    input_format = "RGB"


    with torch.no_grad():
        image_path = "frame000031.jpg"
        #original_image = read_image(image_path, format="BGR")
        original_image = Image.open(image_path)
        resized_image = original_image.resize((640, 640))
        original_image = np.array(resized_image)
        height, width = original_image.shape[:2]
        print("height, width: ", height, width)

        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
        print("image shape: ", image.shape)
        #image = resize_shortest_edge(image, min_size=640, max_size=2560)

        inputs = [{"image": image, "height": height, "width": width}]
        clip_viz_dense = get_lang_feat(inputs, lang_model, is_lang=False)
        #input to HR Module
        high_res_lang = model_super(clip_viz_dense['clip_vis_dense'], clip_viz_dense['res3'], clip_viz_dense['res2'])
        print("high_res_lang shape: ", high_res_lang.shape)

        clip_viz_dense_dim3 = auto_model.encode(high_res_lang)
        print("clip_viz_dense_dim3 shape: ", clip_viz_dense_dim3.shape)
        plt.imshow(clip_viz_dense_dim3[0].permute(1,2,0).detach().cpu().numpy())
        plt.show()
        clip_viz_recon = auto_model.decode(clip_viz_dense_dim3)
        clip_viz_recon_resize = F.interpolate(clip_viz_recon, size=(height, width), mode='bilinear', align_corners=False)

        text_embs = get_user_embed()
        sim_norm_autoencoder = perform_similarity(clip_viz_recon_resize.permute(0,2,3,1), text_embs)
        sim_norm_clip = perform_similarity(high_res_lang.permute(0,2,3,1), text_embs)

        cmap = plt.get_cmap("turbo")
        heatmap_auto = cmap(sim_norm_autoencoder.cpu().numpy())
        heatmap_clip = cmap(sim_norm_clip.cpu().numpy())

        plt.subplot(1,4,1)
        plt.title("Clip heatmap")
        plt.imshow(heatmap_clip)
        plt.subplot(1,4,2)
        plt.title("Encoder heatmap")
        plt.imshow(heatmap_auto)
        plt.subplot(1,4,3)
        plt.title("Dim3 image")
        plt.imshow(clip_viz_dense_dim3[0].permute(1,2,0).cpu().numpy())
        plt.subplot(1,4,4)
        plt.title("Orig. image")
        plt.imshow(original_image)
        plt.show()




