"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import os
import torch
import pdb
from enum import Enum
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte
from torch.utils.data import Dataset
from skimage.color import gray2rgb

from config.serde import read_config



HEIGHT, WIDTH = 299, 299

class Mode(Enum):
    """
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    """
    TRAIN = 0
    TEST = 1
    VALIDATION = 2




class data_loader(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode=Mode.TRAIN):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: enumeration Mode
            Nature of operation to be done with the data.
                Possible inputs are Mode.TRAIN, Mode.VALIDATION, Mode.TEST
                Default value: Mode.TRAIN
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "mimic_master_list.csv"), sep=',')

        if mode==Mode.TRAIN:
            self.subset_df = org_df[org_df['split'] == 'train']
        elif mode == Mode.VALIDATION:
            self.subset_df = org_df[org_df['split'] == 'validate']
        elif mode == Mode.TEST:
            self.subset_df = org_df[org_df['split'] == 'test']

        # choosing a subset due to having large data
        self.chosen_df = self.subset_df[self.subset_df['subset'] == 'p10']

        # self.chosen_df = self.chosen_df[self.chosen_df['subject_id'] == 10000032]
        self.file_path_list = list(self.chosen_df['jpg_rel_path'])


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

        # for this specific model, the images need to have 3 channels
        img = gray2rgb(img) # (h, w, c=3)
        img = resize(img, (HEIGHT, WIDTH)) # (h, w, c=3)

        # Conversion to ubyte value range (0...255) is done here,
        # because network needs to be trained and needs to predict using the same datatype.
        img = img_as_ubyte(img) # (h, w, c=3)

        row = self.chosen_df[self.chosen_df['jpg_rel_path'] == self.file_path_list[idx]]

        label = np.array([int(row['atelectasis'].values[0]), row['cardiomegaly'].values[0], int(row['consolidation']),
                          int(row['edema'].values[0]), int(row['enlarged_cardiomediastinum'].values[0]),
                          int(row['fracture'].values[0]), int(row['lung_lesion'].values[0]),
                          int(row['lung_opacity'].values[0]), int(row['no_finding'].values[0]),
                          int(row['pleural_effusion'].values[0]), int(row['pleural_other'].values[0]),
                          int(row['pneumonia'].values[0]), int(row['pneumothorax'].values[0]),
                          int(row['support_devices'].values[0])]) # (h,)

        # converting the problem to binary multi-label class, by setting everything else than positive, to negative (0)
        label[label != 1] = 0 # (h,)

        img = img.transpose(2, 0, 1)  # (c=3, h, w)
        img = torch.from_numpy(img)  # (c=3, h, w))
        label = torch.from_numpy(label)  # (h,)

        return img, label