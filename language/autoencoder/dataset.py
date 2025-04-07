import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dirs):
        self.data_names = []; get_every_nth_frame = 1
        for data_dir in data_dirs:
            data_list = glob.glob(os.path.join(data_dir, '*.npy'))
            data_list = data_list[::get_every_nth_frame]
            self.data_names.extend(data_list)
            print(f"Loaded {len(self.data_names)} files from {data_dir}")
            

    def __getitem__(self, index):
        features = np.load(self.data_names[index])
        data = torch.tensor(features, dtype=torch.float32)
        #resize to 768x24x24 from 768x192x192
        reshaped_data = F.interpolate(data.unsqueeze(0), size=(24, 24), mode='bilinear', align_corners=False).squeeze(0)
        #F.interpolate(data, size=(24, 24), mode='bilinear', align_corners=False)
        return reshaped_data.float()

    def __len__(self):
        return len(self.data_names)

class Autoencoder_MLP_dataset(Dataset):
    def __init__(self, data_dirs):
        # Combine .npy files from all directories into one list
        self.data_names = []; get_every_nth_frame = 8
        for data_dir in data_dirs:
            data_list = glob.glob(os.path.join(data_dir, '*.npy'))
            data_list = data_list[::get_every_nth_frame]
            self.data_names.extend(data_list)
            print(f"Loaded {len(self.data_names)} files from {data_dir}")
    
    def __getitem__(self, index):
        features = np.load(self.data_names[index])
        data = torch.tensor(features, dtype=torch.float32)
        return data.float()
    
    def __len__(self):
        return len(self.data_names)
