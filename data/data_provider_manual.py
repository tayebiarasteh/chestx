"""
Created on June 22, 2022.
data_provider_manual.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os

import torch
import pandas as pd
import numpy as np
from torchvision import transforms
import cv2
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
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')

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
        self.chosen_labels = ['No finding', 'Aortic enlargement', 'Pleural thickening', 'Cardiomegaly', 'Pleural effusion',
                              'Pneumothorax', 'Atelectasis'] # 7 labels


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
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "CheXpert-v1.0", "master_list.csv"), sep=',')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.subset_df = self.subset_df[self.subset_df['view'] == 'Frontal']
        self.file_path_list = list(self.subset_df['jpg_rel_path'])
        self.chosen_labels = ['cardiomegaly', 'lung_opacity', 'lung_lesion', 'pneumonia', 'edema'] # 5 labels


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
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "master_list.csv"), sep=',')

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
        self.chosen_labels = ['enlarged_cardiomediastinum', 'consolidation', 'pleural_effusion', 'pneumothorax', 'atelectasis'] # 5 labels


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
        self.org_df = pd.read_csv(os.path.join(self.file_base_dir, "final_cxr14_master_list.csv"), sep=',')
        self.file_base_dir = os.path.join(self.file_base_dir, 'CXR14', 'preprocessed')

        if mode == 'train':
            self.subset_df = self.org_df[self.org_df['split'] == 'train']
        elif mode == 'valid':
            self.subset_df = self.org_df[self.org_df['split'] == 'valid']
        elif mode == 'test':
            self.subset_df = self.org_df[self.org_df['split'] == 'test']

        self.file_path_list = list(self.subset_df['img_rel_path'])
        self.chosen_labels = ['cardiomegaly', 'effusion', 'pneumonia', 'consolidation', 'no_finding'] # 5 labels


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
