#!/usr/bin/env python3
"""
Language Feature Visualization Tool

This module provides utilities for extracting, visualizing, and comparing
language features from images at different resolutions.
"""

import os
import sys
import glob
import argparse
import time
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
from tqdm import tqdm
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eval.colormaps import apply_pca_colormap
from supervisedNet import LangSupervisedNet
from load_lang_model import load_lang_model


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
    
    # Normalize to 0-1 range
    sim_norm = (sims - sims.min()) / (sims.max() - sims.min())
    return sim_norm


def get_lang_feat(
    inputs: List[Dict[str, Any]], 
    model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
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


@torch.no_grad()
def plot_lang_heatmap(
    img: torch.Tensor,
    lowres: torch.Tensor,
    highres: torch.Tensor,
    text: str,
    device: str = "cuda",
    save_path: Optional[str] = None
) -> None:
    """
    Plot language feature heatmaps for both low and high resolution features.
    
    Args:
        img: Input image tensor
        lowres: Low-resolution language features
        highres: High-resolution language features
        text: Query text for similarity visualization
        device: Computation device
        save_path: Path to save the visualization (None to display)
    """
    # Get text embeddings and compute similarity
    text_embs = get_user_embed(text=text, device=device)
    sim_norm_low_res = perform_similarity(lowres[0].permute(1, 2, 0), text_embs)
    sim_norm_high_res = perform_similarity(highres[0].permute(1, 2, 0), text_embs)

    # Create heatmap visualizations
    cmap = plt.get_cmap("turbo")
    heatmap_low_res = cmap(sim_norm_low_res.detach().cpu().numpy())
    heatmap_high_res = cmap(sim_norm_high_res.detach().cpu().numpy())

    # Create figure with three subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax[0].imshow(img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Low-resolution heatmap
    ax[1].imshow(heatmap_low_res)
    ax[1].set_title(f"Low-Res Similarity: '{text}'")
    ax[1].axis("off")

    # High-resolution heatmap
    ax[2].imshow(heatmap_high_res)
    ax[2].set_title(f"High-Res Similarity: '{text}'")
    ax[2].axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved heatmap visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


@torch.no_grad()
def plot_features(
    img: torch.Tensor,
    lowres: torch.Tensor,
    highres: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Plot PCA visualizations of language features at different resolutions.
    
    Args:
        img: Input image tensor
        lowres: Low-resolution language features
        highres: High-resolution language features
        save_path: Path to save the visualization (None to display)
    """
    # Create figure with three subplots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax[0].imshow(img.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Low-resolution PCA visualization
    lowres_pca = apply_pca_colormap(lowres.permute(0, 2, 3, 1)).detach().cpu()
    ax[1].imshow(lowres_pca[0])
    ax[1].set_title("Low-Res Feature PCA")
    ax[1].axis("off")
    
    # High-resolution PCA visualization
    highres_pca = apply_pca_colormap(highres.permute(0, 2, 3, 1)).detach().cpu()
    ax[2].imshow(highres_pca[0])
    ax[2].set_title("High-Res Feature PCA")
    ax[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved feature visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def get_hr_feats(
    feat_lang: Dict[str, torch.Tensor],
    model_super: LangSupervisedNet,
    filename: str,
    output_dir: str,
) -> torch.Tensor:
    """
    Generate high-resolution language features and save them if needed.
    
    Args:
        feat_lang: Dictionary of language features from base model
        model_super: High-resolution language model
        filename: Original image filename
        output_dir: Directory to save features
        
    Returns:
        High-resolution language features
    """
    # Generate high-resolution features
    high_res_lang = model_super(
        feat_lang['clip_vis_dense'], 
        feat_lang['res3'], 
        feat_lang['res2']
    )
    
    # Save features if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Save full high-res features
        output_hr_path = os.path.join(output_dir, f"{base_name}_hr.npy")
        np.save(output_hr_path, high_res_lang.detach().cpu().numpy())
        print(f"Saved high-res features to {output_hr_path}")
    
    return high_res_lang


def load_models(
    high_res_checkpoint: str,
    lang_model_path: str,
    device: str = "cuda"
) -> Tuple[LangSupervisedNet, torch.nn.Module, T.Augmentation]:
    """
    Load all required models for feature extraction.
    
    Args:
        high_res_checkpoint: Path to high-resolution model checkpoint
        lang_model_path: Path to language model
        device: Computation device
        
    Returns:
        Tuple containing loaded models and augmentation
    """
    # Load high-resolution model
    model_super = LangSupervisedNet()
    model_super = model_super.load_from_checkpoint(high_res_checkpoint)
    model_super.to(device)
    model_super.eval()
    print("Loaded high-resolution model")

    # Load language model
    model_lang = load_lang_model(model_path=lang_model_path)
    model_lang.to(device)
    model_lang.eval()
    print("Loaded language model")

    # Create augmentation
    aug = T.ResizeShortestEdge([640, 640], 2560)
    
    return model_super, model_lang, aug


def process_image(
    filename: str,
    model_super: LangSupervisedNet,
    model_lang: torch.nn.Module,
    aug: T.Augmentation,
    output_dir: Optional[str] = None,
    query_text: str = "floor",
    device: str = "cuda",
    visualize: bool = True
) -> None:
    """
    Process a single image through the feature extraction pipeline.
    
    Args:
        filename: Path to input image
        model_super: High-resolution language model
        model_lang: Base language model
        aug: Image augmentation transform
        auto_model: Autoencoder model for dimensionality reduction
        output_dir: Directory to save features and visualizations
        query_text: Text query for similarity visualization
        device: Computation device
        visualize: Whether to generate visualizations
    """
    print(f"Processing image: {filename}")
    
    # Read and preprocess image
    original_image = read_image(filename, format="BGR")
    height, width = original_image.shape[:2]
    image = aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    image = image.to(device)
    inputs = [{"image": image, "height": height, "width": width}]
    
    # Extract language features
    start_time = time.time()
    feat_lang = get_lang_feat(inputs, model_lang)
    print(f"Language feature extraction: {time.time() - start_time:.3f}s")
    
    # Generate high-resolution features
    start_time = time.time()
    high_res_lang = get_hr_feats(feat_lang, model_super, filename, output_dir)
    print(f"High-res feature generation: {time.time() - start_time:.3f}s")
    
    # Generate visualizations if requested
    if visualize:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            features_path = os.path.join(output_dir, f"{base_name}_features.png")
            heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap_{query_text}.png")
        else:
            features_path = None
            heatmap_path = None
            
        # Plot feature visualizations
        plot_features(image, feat_lang['clip_vis_dense'], high_res_lang, save_path=features_path)
        
        # Plot language heatmap visualizations
        plot_lang_heatmap(image, feat_lang['clip_vis_dense'], high_res_lang, 
                          text=query_text, device=device, save_path=heatmap_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Language Feature Visualization Tool")
    
    # Model paths
    parser.add_argument("--high-res-model", type=str, required=True,
                        help="Path to high-resolution model checkpoint")
    parser.add_argument("--lang-model", type=str, required=True,
                        help="Path to language model checkpoint")
    
    # Input options
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input image or directory of images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save features and visualizations")
    
    # Visualization options
    parser.add_argument("--query-text", type=str, default="teddybear",
                        help="Text query for similarity visualization")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization generation")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    
    # Load models
    model_super, model_lang, aug = load_models(
        args.high_res_model,
        args.lang_model,
        device
    )
    
    # Process input (single image or directory)
    if os.path.isdir(args.input):
        image_files = sorted(glob.glob(os.path.join(args.input, "*.jpg"))) + \
                     sorted(glob.glob(os.path.join(args.input, "*.png")))
        
        print(f"Found {len(image_files)} images to process")
        
        for img_file in tqdm(image_files, desc="Processing images"):
            process_image(
                img_file,
                model_super,
                model_lang,
                aug,
                auto_model,
                args.output_dir,
                args.query_text,
                device,
                not args.no_visualize
            )
    else:
        # Process single image
        process_image(
            args.input,
            model_super,
            model_lang,
            aug,
            args.output_dir,
            args.query_text,
            device,
            not args.no_visualize
        )
    
    print("Processing complete!")


if __name__ == "__main__":
    main()