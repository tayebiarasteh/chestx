"""
Created on Feb 1, 2022.
data_provider.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os

import matplotlib.pyplot as plt
import torch
import pdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from skimage.util import img_as_ubyte

from config.serde import read_config



epsilon = 1e-15




class vindr_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False):
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
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'vindr-cxr1')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "officialsoroosh_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_officialsoroosh_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed/train')
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed/train')
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']
            self.file_base_dir = os.path.join(self.file_base_dir, 'preprocessed/test')

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

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img)  # (d, h, w)
        # img = img.float() # float32

        label_df = self.subset_df[self.subset_df['image_id'] == self.file_path_list[idx]]
        label = np.array([int(label_df[self.chosen_labels[0]].values[0]), label_df[self.chosen_labels[1]].values[0], int(label_df[self.chosen_labels[2]].values[0]),
                          int(label_df[self.chosen_labels[3]].values[0]), int(label_df[self.chosen_labels[4]].values[0])]) # (h,)
        label = torch.from_numpy(label)  # (h,)
        label = label.float()

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
    def __init__(self, cfg_path, mode='train', augment=False):
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
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'Coronahack_Chest_XRay')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "coronahack_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "officialsoroosh_coronahack_master_list.csv"), sep=',')

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
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])
        img_path = img_path.replace("/Coronahack_Chest_XRay/", "/Coronahack_Chest_XRay/preprocessed/")
        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img)  # (d, h, w)
        # img = img.float() # float32

        label_df = self.subset_df[self.subset_df['X_ray_image_name'] == self.file_path_list[idx]]
        if label_df['Label'].values[0] == 'Normal':
            label = np.array([1, 0, 0])  # (h,)
        else:
            if label_df['Label_1_Virus_category'].values[0] == 'bacteria':
                label = np.array([0, 1, 0])  # (h,)
            elif label_df['Label_1_Virus_category'].values[0] == 'Virus':
                label = np.array([0, 0, 1])  # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

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



class chexpert_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False):
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
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "nothree_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "5000_nothree_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.subset_df = self.subset_df[self.subset_df['view'] == 'Frontal']
        self.file_path_list = list(self.subset_df['jpg_rel_path'])
        self.chosen_labels = ['cardiomegaly', 'lung_opacity', 'lung_lesion', 'pneumonia', 'edema']




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
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])
        img_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed/")
        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img)  # (d, h, w)
        # img = img.float() # float32

        label_df = self.subset_df[self.subset_df['jpg_rel_path'] == self.file_path_list[idx]]
        label = np.array([int(label_df[self.chosen_labels[0]].values[0]), label_df[self.chosen_labels[1]].values[0], int(label_df[self.chosen_labels[2]].values[0]),
                          int(label_df[self.chosen_labels[3]].values[0]), int(label_df[self.chosen_labels[4]].values[0])]) # (h,)

        # setting the label 2 to 0 (negative)
        label[label != 1] = 0 # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

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



class mimic_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False):
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
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, "MIMIC")
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "nothree_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_nothree_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        PAview = self.subset_df[self.subset_df['view'] == 'PA']
        APview = self.subset_df[self.subset_df['view'] == 'AP']
        self.subset_df = PAview.append(APview)
        self.file_path_list = list(self.subset_df['jpg_rel_path'])
        self.chosen_labels = ['enlarged_cardiomediastinum', 'consolidation', 'pleural_effusion', 'pneumothorax', 'atelectasis']



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
        img_path = os.path.join(self.file_base_dir, self.file_path_list[idx])
        img_path = img_path.replace("/files/", "/preprocessed/")
        img = cv2.imread(img_path) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img)  # (d, h, w)
        # img = img.float() # float32

        label_df = self.subset_df[self.subset_df['jpg_rel_path'] == self.file_path_list[idx]]
        label = np.array([int(label_df[self.chosen_labels[0]].values[0]), label_df[self.chosen_labels[1]].values[0], int(label_df[self.chosen_labels[2]].values[0]),
                          int(label_df[self.chosen_labels[3]].values[0]), int(label_df[self.chosen_labels[4]].values[0])]) # (h,)

        # setting the label 2 to 0 (negative)
        label[label != 1] = 0 # (h,)

        label = torch.from_numpy(label)  # (h,)
        label = label.float()

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



class UKA_data_loader_2D(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', augment=False):
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
        self.augment = augment
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'UKA/chest_radiograph')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_UKA_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_final_UKA_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed')
        self.file_path_list = list(self.subset_df['patient_id'])
        self.chosen_labels = ['pleural_effusion_left', 'pleural_effusion_right', 'congestion', 'cardiomegaly', 'pneumonic_infiltrates_left', 'pneumonic_infiltrates_right']



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
        subset = self.subset_df[self.subset_df['patient_id'] == self.file_path_list[idx]]['subset'].values[0]
        img = cv2.imread(os.path.join(self.file_base_dir, subset, str(self.file_path_list[idx]) + '.jpg')) # (h, w, d)

        if self.augment:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomRotation(degrees=10), transforms.ToTensor()])
        else:
            trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        img = trans(img)

        # img = img.transpose(2, 0, 1)
        # img = torch.from_numpy(img)  # (d, h, w)
        # img = img.float() # float32

        label_df = self.subset_df[self.subset_df['patient_id'] == self.file_path_list[idx]]
        if int(label_df[self.chosen_labels[0]].values[0]) < 3:
            first_label = 0
        else:
            first_label = 1
        if int(label_df[self.chosen_labels[1]].values[0]) < 3:
            sec_label = 0
        else:
            sec_label = 1
        if int(label_df[self.chosen_labels[2]].values[0]) < 3:
            third_label = 0
        else:
            third_label = 1
        if int(label_df[self.chosen_labels[3]].values[0]) < 3:
            fourth_label = 0
        else:
            fourth_label = 1
        if int(label_df[self.chosen_labels[4]].values[0]) < 3:
            fifth_label = 0
        else:
            fifth_label = 1
        if int(label_df[self.chosen_labels[5]].values[0]) < 3:
            sixth_label = 0
        else:
            sixth_label = 1
        label = np.array([first_label, sec_label, third_label, fourth_label, fifth_label, sixth_label]) # (h,)
        label = torch.from_numpy(label)  # (h,)
        label = label.float()

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
