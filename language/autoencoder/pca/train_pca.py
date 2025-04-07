import numpy as np
from sklearn.decomposition import PCA
import glob
import os
import joblib

def train_PCA(sample_images):
    # Collect features from sample images
    sample_features = []
    for image_path in sample_images:
        feature_map = np.load(image_path)  # Shape: (768, 192, 192)
        features = feature_map.reshape(768, -1).T  # Shape: (N, 768)
        sample_features.append(features)

    # Concatenate all features
    sample_features = np.concatenate(sample_features, axis=0)

    # Train PCA
    n_components = 23  # Desired compressed dimension
    pca = PCA(n_components=n_components)
    pca.fit(sample_features)

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Save PCA parameters
    pca_mean = pca.mean_
    pca_components = pca.components_
    print(pca_mean.shape, pca_components.shape)
    joblib.dump(pca, "/home/saimouli/Desktop/Bosch/GaussianGripMapping/language/autoencoder/pca/pca_model_23.pkl")
    print("PCA model saved")
    

if __name__ == "__main__":
    hr_feat_dirs = ["/media/saimouli/Data6T/Replica/office1/language_features",
                    "/media/saimouli/Data6T/Replica/room1/language_features",
                    "/media/saimouli/Data6T/Replica/office2/language_features",
                    "/media/saimouli/Data6T/Replica/room2/language_features",
                    "/media/saimouli/Data6T/Replica/office3/language_features",
                    "/media/saimouli/Data6T/Replica/office4/language_features",
                    ]
    data_names = []; get_every_nth_frame = 9
    for data_dir in hr_feat_dirs:
        data_list = glob.glob(os.path.join(data_dir, '*.npy'))
        data_list = data_list[::get_every_nth_frame]
        data_names.extend(data_list)
        print(f"Loaded {len(data_names)} files from {data_dir}")

    #randomy sample 40 paths form sample_images
    #sample_images = np.random.choice(sample_images, 180, replace=False)
    train_PCA(data_names)