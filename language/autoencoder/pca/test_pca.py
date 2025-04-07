import joblib
import glob
import os
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from test_language import perform_similarity, get_user_embed
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import time

def test_PCA(sample_images):
    pca = joblib.load("language/autoencoder/pca/pca_model_23.pkl")
    output_dir = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/language_features/pca_feat_dim24"
    sample_images = sample_images[::10]
    for image_path in sample_images:
        feature_map = np.load(image_path)
        #reshape to 680, 1200
        feature_map_tensor = torch.tensor(feature_map, dtype=torch.float32)
        #feature_map_tensor = F.interpolate(feature_map_tensor.unsqueeze(0), size=(680, 1200), mode='bilinear', align_corners=False).squeeze(0)
        # Calculate the zoom factors for each dimension
        feature_map = feature_map_tensor.numpy()

        H, W = feature_map.shape[1], feature_map.shape[2]

        features = feature_map.reshape(768, -1).T
        #t1 = time.time()
        compressed_features = pca.transform(features)
        #t2 = time.time()
        #print("Time taken for PCA transform (ms): ", (t2-t1)*1000)

        #image_name = os.path.basename(image_path).split(".")[0].split("_")[0]
        #np.save(os.path.join(output_dir, image_name+"_f.npy"), compressed_features.T.reshape(24, H, W))

        reconstructed_features = pca.inverse_transform(compressed_features)
        
        reconstructed_feature_map = reconstructed_features.T.reshape(768, H, W)

        recon_tensor = torch.from_numpy(reconstructed_feature_map).unsqueeze(0)

        text_embs = get_user_embed(device="cpu", text="vase")
        sim_norm = perform_similarity(recon_tensor.permute(0,2,3,1).to("cpu"), text_embs)

        cmap = plt.get_cmap("turbo")
        heatmap = cmap(sim_norm.detach().cpu().numpy())
        plt.imshow(heatmap)
        plt.savefig(os.path.join(output_dir, os.path.basename(image_path).split(".")[0]+"_heatmap.png"))

if __name__ == "__main__":
    hr_feat_dir = "/media/saimouli/RPNG_FLASH_4/datasets/Replica/room0_test/language_features/hr_feat"
    test_images = glob.glob(os.path.join(hr_feat_dir, '*.npy'))
    #test_images = np.random.choice(test_images, 5, replace=False)
    test_PCA(test_images)