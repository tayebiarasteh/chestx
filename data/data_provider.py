"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from config.serde import read_config



epsilon = 1e-15




class vindr_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train'):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'vindr-cxr1/preprocessed')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']
            self.file_base_dir = os.path.join(self.file_base_dir, 'test')

        self.file_path_list = list(self.subset_df['image_id'])
        self.chosen_labels = ['No finding', 'Aortic enlargement', 'Pleural thickening', 'Cardiomegaly', 'Pleural effusion']




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
        img = cv2.imread(os.path.join(self.file_base_dir, self.file_path_list[idx] + '.jpg')) # (h, w, d)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)  # (d, h, w)
        img = img.float() # float32

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]
        label = np.array([int(label_df[self.chosen_labels[0]].values[0]), label_df[self.chosen_labels[1]].values[0], int(label_df[self.chosen_labels[2]].values[0]),
                          int(label_df[self.chosen_labels[3]].values[0]), int(label_df[self.chosen_labels[4]].values[0])]) # (h,)
        label = torch.from_numpy(label)  # (h,)
        label = label.int()

        return img, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['split'] == 'train']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, diseases in enumerate(self.chosen_labels):
            disease_length = sum(train_df[diseases].values == 1)
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class coronahack_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train'):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'Coronahack_Chest_XRay/preprocessed')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "coronahack_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['Dataset_type'] == 'TRAIN']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['Dataset_type'] == 'VALID']
            self.file_base_dir = os.path.join(self.file_base_dir, 'train')
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['Dataset_type'] == 'TEST']
            self.file_base_dir = os.path.join(self.file_base_dir, 'test')

        self.file_path_list = list(self.subset_df['X_ray_image_name'])
        self.chosen_labels = ['Normal', 'bacterial pneumonia', 'viral pneumonia']


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
        img = cv2.imread(os.path.join(self.file_base_dir, self.file_path_list[idx])) # (h, w, d)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)  # (d, h, w)
        img = img.float() # float32

        label_df = self.subset_df[self.subset_df['X_ray_image_name'] == self.file_path_list[idx]]
        if label_df['Label'].values[0] == 'Normal':
            label = np.array([1, 0, 0])  # (h,)
        else:
            if label_df['Label_1_Virus_category'].values[0] == 'bacteria':
                label = np.array([0, 1, 0])  # (h,)
            elif label_df['Label_1_Virus_category'].values[0] == 'Virus':
                label = np.array([0, 0, 1])  # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.int()

        return img, label


    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['Dataset_type'] == 'TRAIN']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        normal_length = len(train_df[train_df['Label'] == 'Normal'])
        bacteria_length = len(train_df[train_df['Label_1_Virus_category'] == 'bacteria'])
        virus_length = len(train_df[train_df['Label_1_Virus_category'] == 'Virus'])
        output_tensor[0] = (full_length - normal_length) / (normal_length + epsilon)
        output_tensor[1] = (full_length - bacteria_length) / (bacteria_length + epsilon)
        output_tensor[2] = (full_length - virus_length) / (virus_length + epsilon)

        return output_tensor



