"""
Created on June 22, 2022.
data_provider_manual.py

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
import random

from config.serde import read_config



epsilon = 1e-15




class vindr_data_loader_2D_manual:
    """
    This is the pipeline based on manual
    """
    def __init__(self, cfg_path, mode='train', augment=False, batch_size=1):
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
        self.batch_size = batch_size
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'vindr-cxr1')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "officialsoroosh_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_officialsoroosh_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "2000_officialsoroosh_master_list.csv"), sep=',')

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
        # self.chosen_labels = ['No finding', 'Aortic enlargement', 'Pleural thickening', 'Cardiomegaly', 'Pleural effusion']
        # self.chosen_labels = ['Cardiomegaly', 'Pleural effusion']
        self.chosen_labels = ['Pleural effusion']


    def provide_mixed(self):
        """training data provider without data augmentation
        Returns
        ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, h, w)
         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, h, w)
        """
        random.shuffle(self.file_path_list)
        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(self.file_path_list, size=self.batch_size, replace=True)

        x_input = []
        y_input = []
        for idx in range(len(chosen_file_list)):
            img = cv2.imread(os.path.join(self.file_base_dir, chosen_file_list[idx] + '.jpg'))  # (h, w, d)

            if self.augment:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=10), transforms.ToTensor()])
            else:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            img = trans(img)

            label_df = self.subset_df[self.subset_df['image_id'] == chosen_file_list[idx]]
            label = torch.zeros((len(self.chosen_labels)))  # (h,)

            for idx in range(len(self.chosen_labels)):
                label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
            label = label.float()
            x_input.append(img)
            y_input.append(label)

        x_input = torch.stack(x_input)  # (n, c=3, h, w)
        y_input = torch.stack(y_input)  # (n, c=3, h, w)

        return x_input, y_input



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



class coronahack_data_loader_2D_manual:
    """
    This is the pipeline on manual
    """
    def __init__(self, cfg_path, mode='train', augment=False, batch_size=1):
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
        self.batch_size = batch_size
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'Coronahack_Chest_XRay')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "coronahack_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "officialsoroosh_coronahack_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "2000_officialsoroosh_coronahack_master_list.csv"), sep=',')

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
        # self.chosen_labels = ['Normal', 'bacteria', 'Virus']
        # self.chosen_labels = ['Normal', 'Pnemonia']
        self.chosen_labels = ['Pnemonia']


    def provide_mixed(self):
        """training data provider without data augmentation
        Returns
        ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, h, w)
         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, h, w)
        """
        random.shuffle(self.file_path_list)
        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(self.file_path_list, size=self.batch_size, replace=True)

        x_input = []
        y_input = []
        for idx in range(len(chosen_file_list)):
            img_path = os.path.join(self.file_base_dir, chosen_file_list[idx])
            img_path = img_path.replace("/Coronahack_Chest_XRay/", "/Coronahack_Chest_XRay/preprocessed/")
            img = cv2.imread(img_path)  # (h, w, d)

            if self.augment:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=10), transforms.ToTensor()])
            else:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            img = trans(img)

            label_df = self.subset_df[self.subset_df['X_ray_image_name'] == chosen_file_list[idx]]

            label = torch.zeros((len(self.chosen_labels)))  # (h,)

            for idx in range(len(self.chosen_labels)):
                if self.chosen_labels[idx] == 'Normal':
                    if label_df['Label'].values[0] == 'Normal':
                        label[idx] = 1
                    else:
                        label[idx] = 0
                elif self.chosen_labels[idx] == 'Pnemonia':
                    if label_df['Label'].values[0] == 'Pnemonia':
                        label[idx] = 1
                    else:
                        label[idx] = 0
                elif self.chosen_labels[idx] == 'bacteria':
                    if label_df['Label_1_Virus_category'].values[0] == 'bacteria':
                        label[idx] = 1
                    else:
                        label[idx] = 0
                elif self.chosen_labels[idx] == 'Virus':
                    if label_df['Label_1_Virus_category'].values[0] == 'Virus':
                        label[idx] = 1
                    else:
                        label[idx] = 0

            label = label.float()
            x_input.append(img)
            y_input.append(label)

        x_input = torch.stack(x_input)  # (n, c=3, h, w)
        y_input = torch.stack(y_input)  # (n, c=3, h, w)

        return x_input, y_input



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        train_df = self.org_df[self.org_df['Dataset_type'] == 'TRAIN']
        full_length = len(train_df)
        output_tensor = torch.zeros((len(self.chosen_labels)))

        for idx, disease in enumerate(self.chosen_labels):
            if disease == 'Normal' or disease == 'Pnemonia':
                disease_length = len(train_df[train_df['Label'] == disease])
            elif disease == 'bacteria' or disease == 'Virus':
                disease_length = len(train_df[train_df['Label_1_Virus_category'] == disease])
            output_tensor[idx] = (full_length - disease_length) / (disease_length + epsilon)

        return output_tensor



class chexpert_data_loader_2D_manual:
    """
    This is the pipeline based on manual
    """
    def __init__(self, cfg_path, mode='train', augment=False, batch_size=1):
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
        self.batch_size = batch_size
        self.file_base_dir = self.params['file_path']
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "nothree_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "5000_nothree_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "2000_nothree_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.subset_df = self.subset_df[self.subset_df['view'] == 'Frontal']
        self.file_path_list = list(self.subset_df['jpg_rel_path'])
        # self.chosen_labels = ['cardiomegaly', 'lung_opacity', 'lung_lesion', 'pneumonia', 'edema']
        self.chosen_labels = ['lung_opacity', 'pneumonia']


    def provide_mixed(self):
        """training data provider without data augmentation
        Returns
        ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, h, w)
         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, h, w)
        """
        random.shuffle(self.file_path_list)
        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(self.file_path_list, size=self.batch_size, replace=True)

        x_input = []
        y_input = []
        for idx in range(len(chosen_file_list)):
            img_path = os.path.join(self.file_base_dir, chosen_file_list[idx])
            img_path = img_path.replace("/CheXpert-v1.0/", "/CheXpert-v1.0/preprocessed/")
            img = cv2.imread(img_path)  # (h, w, d)

            if self.augment:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=10), transforms.ToTensor()])
            else:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            img = trans(img)

            label_df = self.subset_df[self.subset_df['jpg_rel_path'] == chosen_file_list[idx]]
            label = np.zeros((len(self.chosen_labels)))  # (h,)

            for idx in range(len(self.chosen_labels)):
                label[idx] = int(label_df[self.chosen_labels[idx]].values[0])

            # setting the label 2 to 0 (negative)
            label[label != 1] = 0  # (h,)

            label = torch.from_numpy(label)  # (h,)
            label = label.float()
            x_input.append(img)
            y_input.append(label)

        x_input = torch.stack(x_input)  # (n, c=3, h, w)
        y_input = torch.stack(y_input)  # (n, c=3, h, w)

        return x_input, y_input



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



class mimic_data_loader_2D_manual:
    """
    This is the pipeline based on manual
    """
    def __init__(self, cfg_path, mode='train', augment=False, batch_size=1):
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
        self.batch_size = batch_size
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, "MIMIC")
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "nothree_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_nothree_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "2000_nothree_master_list.csv"), sep=',')

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
        # self.chosen_labels = ['enlarged_cardiomediastinum', 'consolidation', 'pleural_effusion', 'pneumothorax', 'atelectasis']
        self.chosen_labels = ['consolidation', 'pleural_effusion']


    def provide_mixed(self):
        """training data provider without data augmentation
        Returns
        ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, h, w)
         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, h, w)
        """
        random.shuffle(self.file_path_list)
        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(self.file_path_list, size=self.batch_size, replace=True)

        x_input = []
        y_input = []
        for idx in range(len(chosen_file_list)):
            img_path = os.path.join(self.file_base_dir, chosen_file_list[idx])
            img_path = img_path.replace("/files/", "/preprocessed/")
            img = cv2.imread(img_path)  # (h, w, d)

            if self.augment:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=10), transforms.ToTensor()])
            else:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            img = trans(img)

            label_df = self.subset_df[self.subset_df['jpg_rel_path'] == chosen_file_list[idx]]
            label = np.zeros((len(self.chosen_labels)))  # (h,)

            for idx in range(len(self.chosen_labels)):
                label[idx] = int(label_df[self.chosen_labels[idx]].values[0])

            # setting the label 2 to 0 (negative)
            label[label != 1] = 0  # (h,)

            label = torch.from_numpy(label)  # (h,)
            label = label.float()
            x_input.append(img)
            y_input.append(label)

        x_input = torch.stack(x_input)  # (n, c=3, h, w)
        y_input = torch.stack(y_input)  # (n, c=3, h, w)

        return x_input, y_input



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



class UKA_data_loader_2D_manual:
    """
    This is the pipeline based on manual
    """
    def __init__(self, cfg_path, mode='train', augment=False, batch_size=1):
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
        self.batch_size = batch_size
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'UKA/chest_radiograph')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_UKA_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_final_UKA_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "2000_final_UKA_master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.file_base_dir = os.path.join(self.file_base_dir, 'UKA_preprocessed')
        self.file_path_list = list(self.subset_df['patient_id'])
        # self.chosen_labels = ['pleural_effusion_left', 'pleural_effusion_right', 'congestion', 'cardiomegaly', 'pneumonic_infiltrates_left', 'pneumonic_infiltrates_right']
        self.chosen_labels = ['pleural_effusion_right', 'pneumonic_infiltrates_left']



    def provide_mixed(self):
        """training data provider without data augmentation
        Returns
        ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, h, w)
         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, h, w)
        """
        random.shuffle(self.file_path_list)
        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(self.file_path_list, size=self.batch_size, replace=True)

        x_input = []
        y_input = []
        for idx in range(len(chosen_file_list)):
            subset = self.subset_df[self.subset_df['patient_id'] == chosen_file_list[idx]]['subset'].values[0]
            img = cv2.imread(
                os.path.join(self.file_base_dir, subset, str(chosen_file_list[idx]) + '.jpg'))  # (h, w, d)

            if self.augment:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=10), transforms.ToTensor()])
            else:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            img = trans(img)

            label_df = self.subset_df[self.subset_df['patient_id'] == chosen_file_list[idx]]

            label = torch.zeros((len(self.chosen_labels)))  # (h,)

            for idx in range(len(self.chosen_labels)):
                if int(label_df[self.chosen_labels[idx]].values[0]) < 3:
                    label[idx] = 0
                else:
                    label[idx] = 1

            label = label.float()
            x_input.append(img)
            y_input.append(label)

        x_input = torch.stack(x_input)  # (n, c=3, h, w)
        y_input = torch.stack(y_input)  # (n, c=3, h, w)

        return x_input, y_input


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



class cxr14_data_loader_2D_manual:
    """
    This is the pipeline based on manual
    """
    def __init__(self, cfg_path, mode='train', augment=False, batch_size=1):
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
        self.batch_size = batch_size
        self.file_base_dir = self.params['file_path']
        self.file_base_dir = os.path.join(self.file_base_dir, 'NIH_ChestX-ray14')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_cxr14_master_list.csv"), sep=',')
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "officialsoroosh_cxr14_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "5000_officialsoroosh_cxr14_master_list.csv"), sep=',')
        # self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "2000_officialsoroosh_cxr14_master_list.csv"), sep=',')
        self.file_base_dir = os.path.join(self.file_base_dir, 'CXR14', 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.file_path_list = list(self.subset_df['img_rel_path'])
        # self.chosen_labels = ['consolidation', 'effusion']
        self.chosen_labels = ['consolidation']


    def provide_mixed(self):
        """training data provider without data augmentation
        Returns
        ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, h, w)
         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, h, w)
        """
        random.shuffle(self.file_path_list)
        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(self.file_path_list, size=self.batch_size, replace=True)

        x_input = []
        y_input = []
        for idx in range(len(chosen_file_list)):
            img = cv2.imread(os.path.join(self.file_base_dir, chosen_file_list[idx]))  # (h, w, d)

            if self.augment:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomRotation(degrees=10), transforms.ToTensor()])
            else:
                trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
            img = trans(img)

            label_df = self.subset_df[self.subset_df['img_rel_path'] == chosen_file_list[idx]]
            label = torch.zeros((len(self.chosen_labels)))  # (h,)

            for idx in range(len(self.chosen_labels)):
                label[idx] = int(label_df[self.chosen_labels[idx]].values[0])
            label = label.float()
            x_input.append(img)
            y_input.append(label)

        x_input = torch.stack(x_input)  # (n, c=3, h, w)
        y_input = torch.stack(y_input)  # (n, c=3, h, w)

        return x_input, y_input



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
