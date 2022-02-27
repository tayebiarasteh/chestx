"""
Created on January 2020.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

from torch.utils.data import Dataset
import torch
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from torchvision import transforms
import os.path
import pdb
import pandas as pd

from config.serde import read_config



train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
epsilon = 1e-15



class ChallengeDataset(Dataset):
    def __init__(self, cfg_path, chosen_labels=[0, 1, 2],
                 transform=transforms.Compose(transforms.ToTensor()), training=True):
        '''
        Args:
            cfg_path (string):
                Config file path of the experiment
            split_ratio (float):
                train-valid splitting
        '''
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.chosen_labels = chosen_labels
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "file.csv"), sep=';')
        self.transform = transform

        if training:
            self.chosen_df = org_df[org_df['subset'] == 'train']
        else:
            self.chosen_df = org_df[org_df['subset'] == 'validate']

        self.file_path_list = list(self.chosen_df['filename'])



    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_path_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """

        img = imread(os.path.join(self.file_base_dir, self.file_path_list[idx])) # (h, w)
        image = gray2rgb(img)
        image = self.transform(image)

        row = self.chosen_df[self.chosen_df['filename'] == self.file_path_list[idx]]

        label = np.array([int(row['poly_wafer'].values[0]), row['crack'].values[0], int(row['inactive'])]) # (h,)

        # choosing the required labels to train with for multi label
        label = np.take(label, self.chosen_labels)
        label = torch.from_numpy(label)

        return image, label


    def pos_weight(self, chosen_labels=[0, 1, 2]):
        '''
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        '''
        full_length = len(self.chosen_df)

        poly_wafer_length = sum(self.chosen_df['poly_wafer'].values == 1)
        w_poly_wafer = torch.tensor((full_length - poly_wafer_length) / (poly_wafer_length + epsilon))

        crack_length = sum(self.chosen_df['crack'].values == 1)
        w_crack = torch.tensor((full_length - crack_length) / (crack_length + epsilon))

        inactive_length = sum(self.chosen_df['inactive'].values == 1)
        w_inactive = torch.tensor((full_length - inactive_length) / (inactive_length + epsilon))

        output_tensor = torch.zeros((3))
        output_tensor[0] = w_poly_wafer
        output_tensor[1] = w_crack
        output_tensor[2] = w_inactive

        output_tensor = np.take(output_tensor, chosen_labels)

        return output_tensor