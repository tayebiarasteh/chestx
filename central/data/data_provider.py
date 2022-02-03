"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import os
import torch
import pdb
import pandas as pd
from torch.utils.data import Dataset
from enum import Enum
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte

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
            self.chosen_df = org_df[org_df['split'] == 'train']
        elif mode == Mode.VALIDATION:
            self.chosen_df = org_df[org_df['split'] == 'validate']
        elif mode == Mode.TEST:
            self.chosen_df = org_df[org_df['split'] == 'test']

        # choosing a subset due to having large data
        self.chosen_df = self.chosen_df[self.chosen_df['subset'] == 'p10']

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
        img = imread(os.path.join(self.file_base_dir, self.file_path_list[idx]))
        img = resize(img, (HEIGHT, WIDTH))

        # Conversion to ubyte value range (0...255) is done here, because network needs to be trained and needs to predict using the same datatype.
        img = img_as_ubyte(img)

        row = self.chosen_df[self.chosen_df['jpg_rel_path'] == self.file_path_list[idx]]

        label = np.array([row['atelectasis'], row['cardiomegaly'], row['consolidation'], row['edema'],
                          row['enlarged_cardiomediastinum'], row['fracture'], row['lung_lesion'], row['lung_opacity'],
                          row['no_finding'], row['pleural_effusion'], row['pleural_other'], row['pneumonia'],
                          row['pneumothorax'], row['support_devices']]) # (h, 1)

        # converting the problem to binary multi-label class, by setting everything else than positive, to negative (0)
        label[label != 1] = 0 # (h, 1)

        img = torch.from_numpy(img)  # (h, w)
        img = torch.unsqueeze(img, 0)  # (c=1, h, w)
        label = torch.from_numpy(label)  # (h, 1)
        label = torch.unsqueeze(label, 0)  # (c=1, h, 1)
        label = torch.squeeze(label, 2)  # (c=1, h)

        return img, label