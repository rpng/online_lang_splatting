import numpy as np
import torch
from load_lang_model import load_lang_model
import detectron2.data.transforms as T
import glob
import os
import sys
from pathlib import Path
from detectron2.data.detection_utils import read_image
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from autoencoder.model import AutoencoderLight
from eval.colormaps import apply_pca_colormap
import matplotlib.pyplot as plt
import open_clip
import time
from typing import Dict, List, Tuple, Union, Any
from supervisedNet import LangSupervisedNet

def get_lang_feat(inputs: List[Dict[str, Any]], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract language features from input images using the provided model.
    
    Args:
        inputs: List of dictionaries containing image data and metadata
        model: Pre-trained language model
        
    Returns:
        Dictionary containing extracted language features
    """
    with torch.no_grad():
        _, dense_clip_viz = model(inputs)
    return dense_clip_viz


def get_user_embed(text: str, device: str = "cuda") -> torch.Tensor:
    """
    Get text embedding using CLIP model.
    
    Args:
        text: Input text string
        device: Computation device
        
    Returns:
        Normalized text embedding tensor
    """
    name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
    tokenizer = open_clip.get_tokenizer(name)
    texts = tokenizer([text]).to(device)
    
    clip_model, _, _ = open_clip.create_model_and_transforms(
        name,
        pretrained=pretrain,
        device=device,
    )
    
    text_embs = clip_model.encode_text(texts)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)
    
    return text_embs


def perform_similarity(
    clip_viz_dense: torch.Tensor, 
    text_embs: torch.Tensor
) -> torch.Tensor:
    """
    Calculate normalized similarity between visual features and text embeddings.
    
    Args:
        clip_viz_dense: Dense visual features from CLIP
        text_embs: Text embeddings
        
    Returns:
        Normalized similarity map
    """
    sims = clip_viz_dense @ text_embs.T
    sims = sims.squeeze()
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    return sim_norm


@torch.no_grad()
def save_hires_labels(
    img_folder: str,
    output_dir: str,
    model_super: None,
    model_lang: torch.nn.Module,
    aug: T.Augmentation,
    auto_model: AutoencoderLight,
    every_n_frame: int = 1,
    vis_interval: int = 100,
    query_text: str = "table"
) -> None:
    """
    Process images and save high-resolution language feature labels.
    
    Args:
        img_folder: Path to folder containing input images
        output_dir: Path to save output files
        model_super: Supervised language model for high-resolution features
        model_lang: Base language model
        aug: Image augmentation transform
        auto_model: Autoencoder model for dimensionality reduction
        every_n_frame: Process every nth frame (default: 1)
        vis_interval: Interval for visualization (default: 100)
        query_text: Text query for similarity visualization (default: "table")
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted image paths and select every nth frame
    img_paths = sorted(glob.glob(os.path.join(img_folder, 'frame*.jpg')))
    img_paths = img_paths[::every_n_frame]
    
    print(f"Processing {len(img_paths)} images from {img_folder}")
    
    # Process each image
    for counter, img_path in enumerate(img_paths):
        img_name = Path(img_path).stem
        print(f"Processing image {counter+1}/{len(img_paths)}: {img_name}")
        
        # Read and preprocess image
        original_image = read_image(img_path, format="BGR")
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image = image.to("cuda")
        inputs = [{"image": image, "height": height, "width": width}]
        
        # Extract language features
        t1 = time.time()
        feat_sed = get_lang_feat(inputs, model_lang)
        fv_sed = feat_sed['clip_vis_dense']
        print(f"Language feature extraction: {time.time() - t1:.3f}s")
        
        # Generate high-resolution language features
        t2 = time.time()
        high_res_lang = model_super(fv_sed, feat_sed['res3'], feat_sed['res2'])
        print(f"High-res feature generation: {time.time() - t2:.3f}s")
        
        # Encode with autoencoder for dimensionality reduction
        N, C, H, W = high_res_lang.shape
        batch_reshape = high_res_lang.permute(0, 2, 3, 1)
        high_res_lang_res = batch_reshape.view(-1, 768)
        
        t3 = time.time()
        low_dim = auto_model.encode(high_res_lang_res)
        print(f"Encoding: {time.time() - t3:.3f}s")
        
        # Decode for visualization (optional)
        recon = auto_model.decode(low_dim)
        recon = recon.view(N, H, W, C).permute(0, 3, 1, 2)
        
        # Generate visualizations at specified intervals
        if counter % vis_interval == 0:
            generate_visualizations(
                img_name,
                output_dir,
                high_res_lang,
                fv_sed,
                recon,
                query_text
            )
        
        # Save low-dimension features
        output_low_dim_path = os.path.join(output_dir, f"{img_name}_ld.npy")
        feature_data = low_dim.T.view(15, 192, 192).detach().cpu().numpy()
        np.save(output_low_dim_path, feature_data)
        
    print(f"Processing complete. Results saved to {output_dir}")


def generate_visualizations(
    img_name: str,
    output_dir: str,
    high_res_lang: torch.Tensor,
    fv_sed: torch.Tensor,
    recon: torch.Tensor,
    query_text: str = "table"
) -> None:
    """
    Generate and save similarity visualizations.
    
    Args:
        img_name: Name of the image
        output_dir: Directory to save visualizations
        high_res_lang: High-resolution language features
        fv_sed: Original language features
        recon: Reconstructed features
        query_text: Text to use for similarity calculation
    """
    text_embs = get_user_embed(device="cuda", text=query_text)
    
    # Calculate similarity maps
    sim_norm_orig = perform_similarity(high_res_lang.permute(0, 2, 3, 1), text_embs)
    sim_norm_sed = perform_similarity(fv_sed.permute(0, 2, 3, 1), text_embs)
    sim_norm_recon = perform_similarity(recon.permute(0, 2, 3, 1), text_embs)
    
    # Create heatmaps
    cmap = plt.get_cmap("turbo")
    heatmap_orig = cmap(sim_norm_orig.detach().cpu().numpy())
    heatmap_sed = cmap(sim_norm_sed.detach().cpu().numpy())
    heatmap_recon = cmap(sim_norm_recon.detach().cpu().numpy())
    
    # Create visualization figure
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original high res ({query_text})")
    plt.imshow(heatmap_orig)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f"SED high res ({query_text})")
    plt.imshow(heatmap_sed)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Reconstructed high res ({query_text})")
    plt.imshow(heatmap_recon)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_similarity.png"), dpi=200)
    plt.close()


def load_models(
    high_res_checkpoint: str,
    lang_model_path: str,
    auto_checkpoint_path: str,
    device: str = "cuda"
) -> Tuple[LangSupervisedNet, torch.nn.Module, T.Augmentation, AutoencoderLight]:
    """
    Load all required models for feature extraction.
    
    Args:
        high_res_checkpoint: Path to high-resolution model checkpoint
        lang_model_path: Path to language model
        auto_checkpoint_path: Path to autoencoder checkpoint
        device: Computation device
        
    Returns:
        Tuple containing loaded models and augmentation
    """
    # Load high-resolution model
    model_super = LangSupervisedNet()
    model_super = model_super.load_from_checkpoint(high_res_checkpoint)
    model_super.to(device)
    model_super.eval()

    # Load language model
    model_lang = load_lang_model(model_path=lang_model_path)
    model_lang.to(device)
    model_lang.eval()

    # Create augmentation
    aug = T.ResizeShortestEdge([640, 640], 2560)

    # Configure and load autoencoder
    encoder_hidden_dims = [384, 192, 96, 48, 24, 15]
    decoder_hidden_dims = [24, 48, 96, 192, 384, 384, 768]
    in_channel_dim = 768
    
    auto_model = AutoencoderLight(
        encoder_hidden_dims, 
        decoder_hidden_dims, 
        in_channel_dim, 
        is_MLP=True
    )
    
    auto_model = auto_model.load_from_checkpoint(
        auto_checkpoint_path, 
        encoder_hidden_dims=encoder_hidden_dims, 
        decoder_hidden_dims=decoder_hidden_dims, 
        in_channel_dim=in_channel_dim,
        is_MLP=True
    )
    auto_model.to(device)
    auto_model.eval()
    
    return model_super, model_lang, aug, auto_model


def main():
    """Main entry point for the script."""
    # Model paths
    check_pt_highres = "supervised/epoch=177-step=39541.ckpt"
    sed_lang_model_path = "seg_clip_model_l.pth"
    auto_ckpt_path = "epoch=133-step=2948.ckpt"
    
    # Load models
    model_super, model_lang, aug, auto_model = load_models(
        check_pt_highres,
        sed_lang_model_path,
        auto_ckpt_path
    )

    # Process each dataset
    datasets = ["room0_test"]
    for dataset in datasets:
        img_folder = f"datasets/Replica/{dataset}/results"
        output_dir = f"Replica/{dataset}/mlp_crossrun1_hr"
        
        print(f"\nProcessing dataset: {dataset}")
        save_hires_labels(
            img_folder,
            output_dir,
            model_super,
            model_lang,
            aug,
            auto_model
        )
        print(f"Completed dataset: {dataset}")


if __name__ == "__main__":
    main()
