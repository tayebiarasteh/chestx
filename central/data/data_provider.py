"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import os
import torch
import pdb
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from torch.utils.data import Dataset
from skimage.color import gray2rgb

from config.serde import read_config



HEIGHT, WIDTH = 299, 299
epsilon = 1e-15




class data_loader(Dataset):
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
        org_df = pd.read_csv(os.path.join(self.file_base_dir, "mimic_master_list.csv"), sep=',')

        if mode=='train':
            self.subset_df = org_df[org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = org_df[org_df['split'] == 'validate']
        elif mode == 'test':
            self.subset_df = org_df[org_df['split'] == 'test']

        # choosing a subset due to having large data
        self.chosen_df1 = self.subset_df[self.subset_df['subset'] == 'p10']
        self.chosen_df2 = self.subset_df[self.subset_df['subset'] == 'p11']

        self.chosen_df = self.chosen_df1.append(self.chosen_df2)

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


    def pos_weight(self):
        '''
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        '''
        full_length = len(self.chosen_df)

        atelectasis_length = sum(self.chosen_df['atelectasis'].values == 1)
        w_atelectasis = torch.tensor((full_length - atelectasis_length) / (atelectasis_length + epsilon))

        cardiomegaly_length = sum(self.chosen_df['cardiomegaly'].values == 1)
        w_cardiomegaly = torch.tensor((full_length - cardiomegaly_length) / (cardiomegaly_length + epsilon))

        consolidation_length = sum(self.chosen_df['consolidation'].values == 1)
        w_consolidation = torch.tensor((full_length - consolidation_length) / (consolidation_length + epsilon))

        edema_length = sum(self.chosen_df['edema'].values == 1)
        w_edema = torch.tensor((full_length - edema_length) / (edema_length + epsilon))

        enlarged_cardiomediastinum_length = sum(self.chosen_df['enlarged_cardiomediastinum'].values == 1)
        w_enlarged_cardiomediastinum = torch.tensor((full_length - enlarged_cardiomediastinum_length) / (enlarged_cardiomediastinum_length + epsilon))

        fracture_length = sum(self.chosen_df['fracture'].values == 1)
        w_fracture = torch.tensor((full_length - fracture_length) / (fracture_length + epsilon))

        lung_lesion_length = sum(self.chosen_df['lung_lesion'].values == 1)
        w_lung_lesion = torch.tensor((full_length - lung_lesion_length) / (lung_lesion_length + epsilon))

        lung_opacity_length = sum(self.chosen_df['lung_opacity'].values == 1)
        w_lung_opacity = torch.tensor((full_length - lung_opacity_length) / (lung_opacity_length + epsilon))

        no_finding_length = sum(self.chosen_df['no_finding'].values == 1)
        w_no_finding = torch.tensor((full_length - no_finding_length) / (no_finding_length + epsilon))

        pleural_effusion_length = sum(self.chosen_df['pleural_effusion'].values == 1)
        w_pleural_effusion = torch.tensor((full_length - pleural_effusion_length) / (pleural_effusion_length + epsilon))

        pleural_other_length = sum(self.chosen_df['pleural_other'].values == 1)
        w_pleural_other = torch.tensor((full_length - pleural_other_length) / (pleural_other_length + epsilon))

        pneumonia_length = sum(self.chosen_df['pneumonia'].values == 1)
        w_pneumonia = torch.tensor((full_length - pneumonia_length) / (pneumonia_length + epsilon))

        pneumothorax_length = sum(self.chosen_df['pneumothorax'].values == 1)
        w_pneumothorax = torch.tensor((full_length - pneumothorax_length) / (pneumothorax_length + epsilon))

        support_devices_length = sum(self.chosen_df['support_devices'].values == 1)
        w_support_devices = torch.tensor((full_length - support_devices_length) / (support_devices_length + epsilon))

        output_tensor = torch.zeros((14))
        output_tensor[0] = w_atelectasis
        output_tensor[1] = w_cardiomegaly
        output_tensor[2] = w_consolidation
        output_tensor[3] = w_edema
        output_tensor[4] = w_enlarged_cardiomediastinum
        output_tensor[5] = w_fracture
        output_tensor[6] = w_lung_lesion
        output_tensor[7] = w_lung_opacity
        output_tensor[8] = w_no_finding
        output_tensor[9] = w_pleural_effusion
        output_tensor[10] = w_pleural_other
        output_tensor[11] = w_pneumonia
        output_tensor[12] = w_pneumothorax
        output_tensor[13] = w_support_devices

        return output_tensor
