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
import glob
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom

from config.serde import read_config


class data_provider_3D():
    def __init__(self, cfg_path, train=True, batch_size=9, valid_batch_size=1):
        """
        Params
        ----------
        """
        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']

        if train:
            self.dataset_path = os.path.join(self.file_base_dir, "train")
            self.batch_size = batch_size
        else:
            self.dataset_path = os.path.join(self.file_base_dir, "valid")
            self.batch_size = valid_batch_size
        self.test_dataset_path = os.path.join(self.file_base_dir, "test")
        self.test_nolabel_dataset_path = os.path.join(self.file_base_dir, "test_no_label")


    def provide_mixed(self):
        '''training data provider without data augmentation

        Returns
        ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, d, h, w)

         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, d, h, w)
        '''
        file_list = glob.glob(os.path.join(self.dataset_path, "*/*"))

        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(file_list, size=self.batch_size, replace=True)

        x_input = []
        y_input = []
        for idx in range(len(chosen_file_list)):
            img = nib.load(chosen_file_list[idx]).get_fdata()
            img = img.astype(np.float32)

            label_dir = chosen_file_list[idx].replace('subvolume-normalized', 'seg-label')
            label_dir = label_dir.replace('/images/', '/labels/')
            label = nib.load(label_dir).get_fdata()
            label = label.astype(np.float32)

            if img.size > 2112000: # 100*160*132 = 2112000
                img, label = self.resize_manual(img, label)

            img = torch.from_numpy(img) # (d, h, w)
            img = torch.unsqueeze(img, 0)  # (c=1, d, h, w)
            x_input.append(img)
            label = torch.from_numpy(label)  # (d, h, w)
            label = torch.unsqueeze(label, 0)  # (c=1, d, h, w)
            y_input.append(label)

        x_input = torch.stack(x_input)  # (n, c=1, d, h, w)
        y_input = torch.stack(y_input)  # (n, c=1, d, h, w)

        return x_input, y_input


    def provide_valid(self):
        '''validation data provider without data augmentation

        Returns
        ----------
         x_input: list of torch tensors of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, d, h, w)

         y_input: list of torch tensors of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, d, h, w)
        '''
        file_list = glob.glob(os.path.join(self.dataset_path, "*/*"))

        x_input = []
        y_input = []
        for idx in range(len(file_list)):
            img = nib.load(file_list[idx]).get_fdata()
            img = img.astype(np.float32)

            label_dir = file_list[idx].replace('subvolume-normalized', 'seg-label')
            label_dir = label_dir.replace('/images/', '/labels/')
            label = nib.load(label_dir).get_fdata()
            label = label.astype(np.float32)

            if img.size > 2112000: # 100*160*132 = 2112000
                img, label = self.resize_manual(img, label)

            img = torch.from_numpy(img) # (d, h, w)
            img = torch.unsqueeze(img, 0)  # (c=1, d, h, w)
            img = img.unsqueeze(0)
            x_input.append(img) # (N, n=1, c=1, d, h, w)
            label = torch.from_numpy(label)  # (d, h, w)
            label = torch.unsqueeze(label, 0)  # (c=1, d, h, w)
            label = label.unsqueeze(0)
            y_input.append(label) # (N, n=1, c=1, d, h, w)

        return x_input, y_input



    def provide_train_mixed_augment(self):
        '''training data provider with data augmentation

         Returns
         ----------
         x_list: torch tensor of float32; n equals the total number of frames per patient
            images
            (n=batch_size, c=1, d, h, w)

         y_list: torch tensor of float32; n equals the total number of frames per patient
            labels
            (n=batch_size, c=1, d, h, w)
         '''
        file_list = glob.glob(os.path.join(self.dataset_path, "*/*"))

        # randomly choose BATCH_SIZE number of images
        chosen_file_list = np.random.choice(file_list, size=self.batch_size, replace=True)

        x_stack = []
        y_stack = []
        for idx in range(len(chosen_file_list)):
            img = nib.load(chosen_file_list[idx]).get_fdata()
            img = img.astype(np.float32)
            x_stack.append(img)

            label_dir = chosen_file_list[idx].replace('subvolume-normalized', 'seg-label')
            label_dir = label_dir.replace('/images/', '/labels/')
            label = nib.load(label_dir).get_fdata()
            y_stack.append(label)

        # final outputs
        x_input = []
        y_input = []

        for i in range(len(x_stack)):
            if x_stack[i].size > 2112000: # 100*160*132 = 2112000
                x, y = self.resize_manual(x_stack[i], y_stack[i])

                # danielle's augmentation
                # y = self.to_categorical(y)
                # augmentor = do_augment(x, y, './config/config.yaml', HVSMRpp=self.HVSMRpp)
                # x_file, y_file = augmentor.apply(x, y)

                # weilin's augmentation
                data = np.zeros(shape=[x.shape[0], x.shape[1], x.shape[2], 2], dtype=np.float32)
                data[:, :, :, 0] = x
                data[:, :, :, 1] = y
                trans_data = random_augmentation(data, './config/config.yaml')
                x_file = trans_data[:, :, :, 0]
                y_file = trans_data[:, :, :, 1]

            else:
                # danielle's augmentation
                # y = self.to_categorical(y)
                # augmentor = do_augment(x, y, './config/config.yaml', HVSMRpp=self.HVSMRpp)
                # x_file, y_file = augmentor.apply(x, y)

                # weilin's augmentation
                data = np.zeros(shape=[x_stack[i].shape[0], x_stack[i].shape[1], x_stack[i].shape[2], 2], dtype=np.float32)
                data[:, :, :, 0] = x_stack[i]
                data[:, :, :, 1] = y_stack[i]
                trans_data = random_augmentation(data, './config/config.yaml')
                x_file = trans_data[:, :, :, 0]
                y_file = trans_data[:, :, :, 1]

            x_file = torch.from_numpy(x_file)  # (d, h, w)
            x_file = torch.unsqueeze(x_file, 0)  # (c=1, d, h, w)
            y_file = y_file.astype(np.int8)
            y_file = torch.from_numpy(y_file)  # (d, h, w)
            y_file = torch.unsqueeze(y_file, 0)  # (c=1, d, h, w)
            x_input.append(x_file)
            y_input.append(y_file)
        x_input = torch.stack(x_input)  # (n, c=1, d, h, w)
        y_input = torch.stack(y_input)  # (n, c=1, d, h, w)

        return x_input, y_input


    def provide_test_with_label(self):
        '''test data provider with labels

        Returns
        ----------
        full_x_input: list of torch tensors of float32; n equals the total number of frames per patient
                            N equals the total number of patients
            images
            (N=pat_num, n=30, c=1, d, h, w)

        full_y_input: list of torch tensors of int8; n equals the total number of frames per patient
                            N equals the total number of patients
            labels
            (N=pat_num, n=30, c=1, d, h, w)
        '''
        patient_ls = os.listdir(self.test_dataset_path)
        patient_ls.sort()
        full_x_input = []
        full_y_input = []

        for patient in patient_ls:
            im_dir = os.path.join(self.test_dataset_path, patient)
            test_file_list = glob.glob(im_dir + '/*.nii.gz')
            test_file_list.sort()

            x_input = []
            y_input = []
            for idx in range(len(test_file_list)):
                img = nib.load(test_file_list[idx]).get_fdata()

                img = img.astype(np.float32)
                x_input.append(img)
                label_dir = test_file_list[idx].replace('subvolume-normalized', 'seg-label')
                label_dir = label_dir.replace('/images/', '/labels/')
                label = nib.load(label_dir).get_fdata()
                label = label.astype(np.float32)
                y_input.append(label)

            x_input = np.stack(x_input)
            x_input = torch.from_numpy(x_input) # (n=30, d, h, w)
            x_input = torch.unsqueeze(x_input, 1) # (n=30, c=1, d, h, w)
            y_input = np.stack(y_input)
            y_input = y_input.astype(np.int8)
            y_input = torch.from_numpy(y_input) # (n=30, d, h, w)
            y_input = torch.unsqueeze(y_input, 1) # (n=30, c=1, d, h, w)
            full_x_input.append(x_input) # (N=pat_num, n=30, c=1, d, h, w)
            full_y_input.append(y_input) # (N=pat_num, n=30, c=1, d, h, w)

        return full_x_input, full_y_input



    def provide_test_without_label(self):
        '''test data provider with labels

        Returns
        ----------
        full_x_input: list of torch tensors; n equals the total number of frames per patient
                            N equals the total number of patients
            images
            (N=pat_num, n=30, c=1, d, h, w)

        full_x_input_nifti: list of frames of list of patients
                            nifti image with header and information
            original cropped normalized images (d, h, w)

        full_image_names: list of frames of list of patients
                            str
            names of the original cropped normalized images
        '''
        patient_ls = os.listdir(self.test_nolabel_dataset_path)
        patient_ls.sort()
        full_x_input = []
        full_x_input_nifti = []
        full_image_names = []

        for patient in patient_ls:
            im_dir = os.path.join(self.test_nolabel_dataset_path, patient)
            test_file_list = glob.glob(im_dir + '/*.nii.gz')
            test_file_list.sort()

            x_input = []
            x_input_nifti = []
            image_names = []
            for idx in range(len(test_file_list)):
                img_nifti = nib.load(test_file_list[idx])
                img = img_nifti.get_fdata()
                img = img.astype(np.float32)
                x_input.append(img)
                x_input_nifti.append(img_nifti)
                image_names.append(os.path.basename(test_file_list[idx]))

            x_input = np.stack(x_input)
            x_input = torch.from_numpy(x_input) # (n=30, d, h, w)
            x_input = torch.unsqueeze(x_input, 1) # (n=30, c=1, d, h, w)
            full_x_input.append(x_input) # (N=pat_num, n=30, c=1, d, h, w)
            full_x_input_nifti.append(x_input_nifti)
            full_image_names.append(image_names)

        return full_x_input, full_x_input_nifti, full_image_names


    def to_categorical(self, input_gt, num_classes=9):
        '''
        from keras to categorical
        making one-hot encoding
        '''
        y = np.array(input_gt, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype='float32')
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        output_gt = np.reshape(categorical, output_shape)

        return output_gt


    def resize_manual(self, img, gt):
        resize_ratio = np.divide(tuple(self.params['augmentation_weilin']['resize_shape']), img.shape)
        img = zoom(img, resize_ratio, order=2)
        gt = zoom(gt, resize_ratio, order=0)
        return img, gt



if __name__=='__main__':
    CONFIG_PATH = '../config/config.yaml'
    data_samples = data_provider_3D(cfg_path=CONFIG_PATH)
    x_input, y_input = data_samples.provide_train_mixed_augment()
    # x_input, y_input = data_samples.provide_test_with_label()
    # x_input, y_input = data_samples.provide_train_mixed_augment()
    print(x_input.shape)
    print(y_input.shape)
    pdb.set_trace()
    sgdsfg=0