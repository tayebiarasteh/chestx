"""
Created on Feb 2, 2022.
csv_data_preprocess.py

data preprocessing for X-Ray images.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""
import glob
import os
import pdb
import pandas as pd
from tqdm import tqdm
import pydicom as dicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from scipy.ndimage.interpolation import zoom
import random

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')

HEIGHT, WIDTH = 512, 512



class csv_preprocess_mimic():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"):
        self.params = read_config(cfg_path)


    def csv_creator(self):
        """csv_creator"""

        output_path = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list.csv"
        output_path1 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list1.csv"
        output_path2 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list2.csv"
        output_path3 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list3.csv"
        output_path4 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list4.csv"
        output_path5 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list5.csv"
        output_path6 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list6.csv"
        output_path7 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list7.csv"
        output_path8 = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list8.csv"

        metda_data_path = "/home/soroosh/Documents/datasets/MIMIC/mimic-cxr-2.0.0-metadata.csv"
        metda_data_df = pd.read_csv(metda_data_path, sep=',')

        chexpert_path = "/home/soroosh/Documents/datasets/MIMIC/mimic-cxr-2.0.0-chexpert.csv"
        chexpert_df = pd.read_csv(chexpert_path, sep=',')

        split_path = "/home/soroosh/Documents/datasets/MIMIC/mimic-cxr-2.0.0-split.csv"
        split_df = pd.read_csv(split_path, sep=',')

        study_path = "/home/soroosh/Documents/datasets/MIMIC/cxr-study-list.csv"
        study_df = pd.read_csv(study_path, sep=',')

        record_path = "/home/soroosh/Documents/datasets/MIMIC/cxr-record-list.csv"
        record_df = pd.read_csv(record_path, sep=',')

        final_data = pd.DataFrame(
            columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view', 'available_views',
                     'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
                     'enlarged_cardiomediastinum', 'fracture', 'lung_lesion', 'lung_opacity', 'no_finding',
                     'pleural_effusion', 'pleural_other', 'pneumonia', 'pneumothorax', 'support_devices'])

        for index, row in tqdm(metda_data_df.iterrows()):
            jpg_rel_path = os.path.join('mimic-cxr-jpg', (record_df[record_df['dicom_id'] ==
                                                                    row['dicom_id']]['path'].values[0]).replace('.dcm',
                                                                                                                '.jpg'))
            report_rel_path = os.path.join('mimic-cxr-reports',
                                           (study_df[study_df['study_id'] == row['study_id']]['path'].values[0]))
            subject_id = row['subject_id']
            study_id = row['study_id']
            try:
                split = split_df[split_df['dicom_id'] == row['dicom_id']]['split'].values[0]
            except:
                split = 'train'
            view = row['ViewPosition']
            available_views = row['PerformedProcedureStepDescription']
            n_x_pixels = row['Rows']
            n_y_pixels = row['Columns']

            try:
                atelectasis = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Atelectasis'].values[0]
            except:
                atelectasis = float("NAN")
            try:
                cardiomegaly = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Cardiomegaly'].values[0]
            except:
                cardiomegaly = float("NAN")
            try:
                consolidation = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Consolidation'].values[0]
            except:
                consolidation = float("NAN")
            try:
                edema = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Edema'].values[0]
            except:
                edema = float("NAN")
            try:
                enlarged_cardiomediastinum = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Enlarged Cardiomediastinum'].values[0]
            except:
                enlarged_cardiomediastinum = float("NAN")
            try:
                fracture = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Fracture'].values[0]
            except:
                fracture = float("NAN")
            try:
                lung_lesion = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Lung Lesion'].values[0]
            except:
                lung_lesion = float("NAN")
            try:
                lung_opacity = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Lung Opacity'].values[0]
            except:
                lung_opacity = float("NAN")
            try:
                no_finding = chexpert_df[chexpert_df['study_id'] == row['study_id']]['No Finding'].values[0]
            except:
                no_finding = float("NAN")
            try:
                pleural_effusion = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Pleural Effusion'].values[0]
            except:
                pleural_effusion = float("NAN")
            try:
                pleural_other = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Pleural Other'].values[0]
            except:
                pleural_other = float("NAN")
            try:
                pneumonia = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Pneumonia'].values[0]
            except:
                pneumonia = float("NAN")
            try:
                pneumothorax = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Pneumothorax'].values[0]
            except:
                pneumothorax = float("NAN")
            try:
                support_devices = chexpert_df[chexpert_df['study_id'] == row['study_id']]['Support Devices'].values[0]
            except:
                support_devices = float("NAN")

            tempp = pd.DataFrame([[jpg_rel_path, report_rel_path, subject_id, study_id, split, view, available_views,
                                   n_x_pixels, n_y_pixels, atelectasis, cardiomegaly, consolidation, edema,
                                   enlarged_cardiomediastinum, fracture, lung_lesion, lung_opacity, no_finding,
                                   pleural_effusion, pleural_other, pneumonia, pneumothorax, support_devices]],
                                 columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                                          'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                                          'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture',
                                          'lung_lesion', 'lung_opacity', 'no_finding', 'pleural_effusion',
                                          'pleural_other', 'pneumonia', 'pneumothorax', 'support_devices'])
            final_data = final_data.append(tempp)
            # sort based on name
            if (index + 1) == 50000:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path1, sep=',', index=False)
                del final_data
                final_data = pd.DataFrame(
                    columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                             'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                             'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                             'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
                             'pneumothorax', 'support_devices'])

            elif (index + 1) == 100000:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path2, sep=',', index=False)
                del final_data
                final_data = pd.DataFrame(
                    columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                             'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                             'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                             'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
                             'pneumothorax', 'support_devices'])

            elif (index + 1) == 150000:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path3, sep=',', index=False)
                del final_data
                final_data = pd.DataFrame(
                    columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                             'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                             'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                             'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
                             'pneumothorax', 'support_devices'])

            elif (index + 1) == 200000:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path4, sep=',', index=False)
                del final_data
                final_data = pd.DataFrame(
                    columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                             'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                             'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                             'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
                             'pneumothorax', 'support_devices'])

            elif (index + 1) == 250000:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path5, sep=',', index=False)
                del final_data
                final_data = pd.DataFrame(
                    columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                             'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                             'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                             'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
                             'pneumothorax', 'support_devices'])

            elif (index + 1) == 300000:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path6, sep=',', index=False)
                del final_data
                final_data = pd.DataFrame(
                    columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                             'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                             'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                             'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
                             'pneumothorax', 'support_devices'])

            elif (index + 1) == 350000:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path7, sep=',', index=False)
                del final_data
                final_data = pd.DataFrame(
                    columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view',
                             'available_views', 'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly',
                             'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture', 'lung_lesion',
                             'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
                             'pneumothorax', 'support_devices'])

        # sort based on name
        final_data = final_data.sort_values(['jpg_rel_path'])
        final_data.to_csv(output_path8, sep=',', index=False)

        # merging the dataframes
        df1 = pd.read_csv(output_path1, sep=',')
        df2 = pd.read_csv(output_path2, sep=',')
        df3 = pd.read_csv(output_path3, sep=',')
        df4 = pd.read_csv(output_path4, sep=',')
        df5 = pd.read_csv(output_path5, sep=',')
        df6 = pd.read_csv(output_path6, sep=',')
        df7 = pd.read_csv(output_path7, sep=',')
        ultimate_df = df1.append(df2)
        ultimate_df = ultimate_df.append(df3)
        ultimate_df = ultimate_df.append(df4)
        ultimate_df = ultimate_df.append(df5)
        ultimate_df = ultimate_df.append(df6)
        ultimate_df = ultimate_df.append(df7)
        ultimate_df = ultimate_df.append(final_data)

        # assigning the subset number
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p10', 'subset'] = 'p10'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p11', 'subset'] = 'p11'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p12', 'subset'] = 'p12'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p13', 'subset'] = 'p13'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p14', 'subset'] = 'p14'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p15', 'subset'] = 'p15'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p16', 'subset'] = 'p16'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p17', 'subset'] = 'p17'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p18', 'subset'] = 'p18'
        ultimate_df.loc[ultimate_df.jpg_rel_path.str.slice(20, 23) == 'p19', 'subset'] = 'p19'

        ultimate_df.to_csv(output_path, sep=',', index=False)


    def class_num_change(self):
        """
        Class 0 will stay 0: "negative"
        Class 1 will stay 1: "positive"
        Class -1 will become class 3: "uncertain positive"
        Class NaN will become class 2: not given; not mentioned in the report
        """

        output_path = "/home/soroosh/Documents/datasets/XRay/MIMIC/mimic_master_list.csv"
        newoutput_path = "/home/soroosh/Documents/datasets/XRay/MIMIC/newmimic_master_list.csv"

        df1 = pd.read_csv(output_path, sep=',')

        df1[['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture',
             'lung_lesion', 'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
             'pneumothorax', 'support_devices']] = df1[
            ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture',
             'lung_lesion', 'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
             'pneumothorax', 'support_devices']].fillna(5).astype(int)

        df1.loc[df1.atelectasis == -1, 'atelectasis'] = 3
        df1.loc[df1.cardiomegaly == -1, 'cardiomegaly'] = 3
        df1.loc[df1.consolidation == -1, 'consolidation'] = 3
        df1.loc[df1.edema == -1, 'edema'] = 3
        df1.loc[df1.enlarged_cardiomediastinum == -1, 'enlarged_cardiomediastinum'] = 3
        df1.loc[df1.fracture == -1, 'fracture'] = 3
        df1.loc[df1.lung_lesion == -1, 'lung_lesion'] = 3
        df1.loc[df1.lung_opacity == -1, 'lung_opacity'] = 3
        df1.loc[df1.no_finding == -1, 'no_finding'] = 3
        df1.loc[df1.pleural_effusion == -1, 'pleural_effusion'] = 3
        df1.loc[df1.pleural_other == -1, 'pleural_other'] = 3
        df1.loc[df1.pneumonia == -1, 'pneumonia'] = 3
        df1.loc[df1.pneumothorax == -1, 'pneumothorax'] = 3
        df1.loc[df1.support_devices == -1, 'support_devices'] = 3

        df1.loc[df1.atelectasis == 5, 'atelectasis'] = 2
        df1.loc[df1.cardiomegaly == 5, 'cardiomegaly'] = 2
        df1.loc[df1.consolidation == 5, 'consolidation'] = 2
        df1.loc[df1.edema == 5, 'edema'] = 2
        df1.loc[df1.enlarged_cardiomediastinum == 5, 'enlarged_cardiomediastinum'] = 2
        df1.loc[df1.fracture == 5, 'fracture'] = 2
        df1.loc[df1.lung_lesion == 5, 'lung_lesion'] = 2
        df1.loc[df1.lung_opacity == 5, 'lung_opacity'] = 2
        df1.loc[df1.no_finding == 5, 'no_finding'] = 2
        df1.loc[df1.pleural_effusion == 5, 'pleural_effusion'] = 2
        df1.loc[df1.pleural_other == 5, 'pleural_other'] = 2
        df1.loc[df1.pneumonia == 5, 'pneumonia'] = 2
        df1.loc[df1.pneumothorax == 5, 'pneumothorax'] = 2
        df1.loc[df1.support_devices == 5, 'support_devices'] = 2

        df1.to_csv(newoutput_path, sep=',', index=False)


    def threetwo_remover(self):
        path = "/home/soroosh/Documents/datasets/XRay/MIMIC/mimic_master_list.csv"
        newoutput_path = "/home/soroosh/Documents/datasets/XRay/MIMIC/nothree_master_list.csv"

        final_data = pd.DataFrame(columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view', 'available_views',
                     'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
                     'enlarged_cardiomediastinum', 'fracture', 'lung_lesion', 'lung_opacity', 'no_finding',
                     'pleural_effusion', 'pleural_other', 'pneumonia', 'pneumothorax', 'support_devices', 'subset'])

        df = pd.read_csv(path, sep=',')
        for index, row in tqdm(df.iterrows()):
            # if (row['enlarged_cardiomediastinum'] < 2) and (row['consolidation'] < 2) and (row['pleural_effusion'] < 2) and (row['pneumothorax'] < 2) and (row['atelectasis'] < 2):
            if (row['enlarged_cardiomediastinum'] < 3) and (row['consolidation'] < 3) and (row['pleural_effusion'] < 3) and (row['pneumothorax'] < 3) and (row['atelectasis'] < 3):
            # if (row['cardiomegaly'] < 2):
                final_data = final_data.append(row)

        final_data.to_csv(newoutput_path, sep=',', index=False)



class normalizer_resizer():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"):
        pass


    def mimic_normalizer_resizer(self):
        base_path = '/home/soroosh/Documents/datasets/MIMIC'

        df_path = os.path.join(base_path, 'mimic_master_list.csv')
        df = pd.read_csv(df_path, sep=',')

        file_list = df['jpg_rel_path'].to_list()
        for file_path in tqdm(file_list):

            image_path = os.path.join(base_path, file_path)
            image = cv2.imread(image_path)

            # color to gray
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize
            resize_ratio = np.divide((HEIGHT, WIDTH), img.shape)
            img = zoom(img, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = image_path.replace('/files/', '/preprocessed/')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, img)



    def vindr_normalizer_resizer(self):

        path = "/home/soroosh/Documents/datasets/vindr-cxr1/original"

        train_file_list = glob.glob(os.path.join(path, 'train/*.dicom'))
        for file_path in tqdm(train_file_list):

            RefDs = dicom.dcmread(file_path)

            img = RefDs.pixel_array

            # resize
            resize_ratio = np.divide((HEIGHT, WIDTH), img.shape)
            img = zoom(img, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # invert the values if necessary
            if RefDs[0x0028, 0x0004].value == 'MONOCHROME1':
                img = np.invert(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = file_path.replace('/original/', '/preprocessed/')
            output_path = output_path.replace('.dicom', '.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, img)


        test_file_list = glob.glob(os.path.join(path, 'test/*.dicom'))
        for file_path in test_file_list:

            RefDs = dicom.dcmread(file_path)

            img = RefDs.pixel_array

            resize_ratio = np.divide((HEIGHT, WIDTH), img.shape)
            img = zoom(img, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # invert the values if necessary
            if RefDs[0x0028, 0x0004].value == 'MONOCHROME1':
                img = np.invert(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = file_path.replace('/original/', '/preprocessed/')
            output_path = output_path.replace('.dicom', '.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, img)



    def vindrpediatric_normalizer_resizer(self):

        path = "/home/soroosh/Documents/datasets/vindr-pcxr/original"

        train_file_list = glob.glob(os.path.join(path, 'train/*.dicom'))
        for file_path in tqdm(train_file_list):

            RefDs = dicom.dcmread(file_path)

            img = RefDs.pixel_array

            # resize
            resize_ratio = np.divide((HEIGHT, WIDTH), img.shape)
            img = zoom(img, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # invert the values if necessary
            if RefDs[0x0028, 0x0004].value == 'MONOCHROME1':
                img = np.invert(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = file_path.replace('/original/', '/preprocessed/')
            output_path = output_path.replace('.dicom', '.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, img)


        test_file_list = glob.glob(os.path.join(path, 'test/*.dicom'))
        for file_path in test_file_list:

            RefDs = dicom.dcmread(file_path)

            img = RefDs.pixel_array

            resize_ratio = np.divide((HEIGHT, WIDTH), img.shape)
            img = zoom(img, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # invert the values if necessary
            if RefDs[0x0028, 0x0004].value == 'MONOCHROME1':
                img = np.invert(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = file_path.replace('/original/', '/preprocessed/')
            output_path = output_path.replace('.dicom', '.jpg')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, img)



    def chexpert_normalizer_resizer(self):
        base_path = '/mnt/hdd/Share/Chexpert_dataset/'
        valid_path = os.path.join(base_path, 'CheXpert-v1.0/valid.csv')
        valid_df = pd.read_csv(valid_path, sep=',')

        valid_file_list = valid_df['Path'].to_list()
        for file_path in tqdm(valid_file_list):

            image_path = os.path.join(base_path, file_path)
            image = cv2.imread(image_path)

            # color to gray
            src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize
            resize_ratio = np.divide((HEIGHT, WIDTH), src.shape)
            img = zoom(src, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = image_path.replace('/CheXpert-v1.0/', '/CheXpert-v1.0/preprocessed/')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)


        train_path = os.path.join(base_path, 'CheXpert-v1.0/train.csv')
        train_df = pd.read_csv(train_path, sep=',')

        train_file_list = train_df['Path'].to_list()
        for file_path in tqdm(train_file_list):

            image_path = os.path.join(base_path, file_path)
            image = cv2.imread(image_path)

            # color to gray
            src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize
            resize_ratio = np.divide((HEIGHT, WIDTH), src.shape)
            img = zoom(src, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = image_path.replace('/CheXpert-v1.0/', '/CheXpert-v1.0/preprocessed/')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)



    def pediatric_corona_normalizer_resizer(self):
        base_path = '/home/soroosh/Documents/datasets/Coronahack_Chest_XRay/original'

        df_path = os.path.join(base_path, 'Chest_xray_Corona_Metadata.csv')
        df = pd.read_csv(df_path, sep=',')

        for index, row in tqdm(df.iterrows()):

            if row['Dataset_type'] == 'TRAIN':
                image_path = os.path.join(base_path, 'train', row['X_ray_image_name'])
            else:
                image_path = os.path.join(base_path, 'test', row['X_ray_image_name'])

            image = cv2.imread(image_path)

            # color to gray
            src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # resize
            resize_ratio = np.divide((HEIGHT, WIDTH), src.shape)
            img = zoom(src, resize_ratio, order=2)

            # normalization
            min_ = np.min(img)
            max_ = np.max(img)
            scale = max_ - min_
            img = (img - min_) / scale

            # converting to the range [0 255]
            img = img_as_ubyte(img)

            # histogram equalization
            img = cv2.equalizeHist(img)
            output_path = image_path.replace('/original/', '/preprocessed/')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            cv2.imwrite(output_path, img)



    def UKA_normalizer_resizer(self):

        base_path = "/data/chest_radiograph/dicom_files"
        flag = 0
        final_df = pd.DataFrame(columns=['image_id', 'split', 'subset', 'birth_date', 'examination_date', 'study_time',
                                            'patient_sex', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',	'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        label_path = '/data/chest_radiograph/UKA_master_list.csv'
        final_df_output_path = '/data/chest_radiograph/final_UKA_master_list.csv'
        df = pd.read_csv(label_path, sep=',')
        counter = 0
        subset_num = 1

        file_list = glob.glob(os.path.join(base_path, '*/*/*'))

        for file_path in tqdm(file_list):

            RefDs = dicom.dcmread(file_path)

            try:
                img = RefDs.pixel_array

                # resize
                resize_ratio = np.divide((HEIGHT, WIDTH), img.shape)
                img = zoom(img, resize_ratio, order=2)

                # normalization
                min_ = np.min(img)
                max_ = np.max(img)
                scale = max_ - min_
                img = (img - min_) / scale

                # converting to the range [0 255]
                img = img_as_ubyte(img)

                # invert the values if necessary
                if RefDs[0x0028, 0x0004].value == 'MONOCHROME1':
                    img = np.invert(img)

                # histogram equalization
                img = cv2.equalizeHist(img)
                output_path = file_path.replace('/dicom_files/', '/preprocessed/')
                basename1 = os.path.basename(output_path)
                basename2 = os.path.basename(os.path.dirname(output_path))
                patient_id = os.path.basename(os.path.dirname(os.path.dirname(output_path)))

                output_path = output_path.replace('/' + basename1, '')
                output_path = output_path.replace('/' + basename2, '')

                chosen_df = df[df['image_id'] == int(patient_id)]
                try:
                    if chosen_df['split'].values[0] == 'test':
                        subset = 'test'
                        output_path = output_path.replace(patient_id, subset + '/' + patient_id + '.jpg')

                    if chosen_df['split'].values[0] == 'valid':
                        subset = 'valid'
                        output_path = output_path.replace(patient_id, subset + '/' + patient_id + '.jpg')

                    if chosen_df['split'].values[0] == 'train':
                        counter += 1
                        if counter < 12000:
                            subset = 'p' + str(int(subset_num))
                            output_path = output_path.replace(patient_id, subset + '/' + patient_id + '.jpg')
                        else:
                            counter = 0
                            subset_num += 1
                            subset = 'p' + str(int(subset_num))
                            output_path = output_path.replace(patient_id, subset + '/' + patient_id + '.jpg')
                except:
                    continue

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                cv2.imwrite(output_path, img)
                tempp = pd.DataFrame(
                    [[chosen_df['image_id'].values[0], chosen_df['split'].values[0], subset, chosen_df['birth_date'].values[0], chosen_df['examination_date'].values[0], chosen_df['study_time'].values[0],
                      chosen_df['patient_sex'].values[0], chosen_df['ExposureinuAs'].values[0], chosen_df['cardiomegaly'].values[0], chosen_df['congestion'].values[0], chosen_df['pleural_effusion_right'].values[0],
                             chosen_df['pleural_effusion_left'].values[0],
                             chosen_df['pneumonic_infiltrates_right'].values[0], chosen_df['pneumonic_infiltrates_left'].values[0], chosen_df['disturbances_right'].values[0],
                             chosen_df['disturbances_left'].values[0], chosen_df['pneumothorax_right'].values[0], chosen_df['pneumothorax_left'].values[0], chosen_df['subject_id'].values[0]]],
                    columns=['image_id', 'split', 'subset', 'birth_date', 'examination_date', 'study_time',
                             'patient_sex', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right',
                             'pleural_effusion_left',
                             'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',
                             'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])
                final_df = final_df.append(tempp)
                final_df.to_csv(final_df_output_path, sep=',', index=False)

            except:
                flag += 1
                print(flag, file_path)


    def UKA_csvfixer(self):
        path_org = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/org_UKA_master_list.csv'
        org_df = pd.read_csv(path_org, sep=',', low_memory=False)

        path_4k = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/final_UKA_master_list.csv'
        small4k_df = pd.read_csv(path_4k, sep=',', low_memory=False)

        path_deutsch = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/original_labels/train_valid_test.csv'
        deutsch_df = pd.read_csv(path_deutsch, sep=',', low_memory=False)

        file_list = glob.glob('/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/UKA_preprocessed/*/*.jpg')

        final_path_org = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/original_UKA_master_list.csv'
        final_df_org = pd.DataFrame(
            columns=['image_id', 'split', 'subset', 'birth_date', 'examination_date', 'study_time', 'patient_sex',
                     'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',
                     'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        final_path_4K = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/final_test4k_UKA_master_list.csv'
        final_df_4K = pd.DataFrame(
            columns=['image_id', 'split', 'subset', 'birth_date', 'examination_date', 'study_time', 'patient_sex',
                     'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',
                     'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        file_list.sort()

        for img in tqdm(file_list):
            image_id = os.path.basename(img).replace('.jpg', '')
            chosen_df = org_df[org_df['image_id'] == int(image_id)]
            subject = deutsch_df[deutsch_df['Anforderungsnummer'] == int(image_id)]['Aufnahmenummer']
            chosen_df = chosen_df.assign(subject_id=subject.values.item(0))
            final_df_org = final_df_org.append(chosen_df[0:1])
            final_df_org.to_csv(final_path_org, sep=',', index=False)


        for img in tqdm(file_list):
            image_id = os.path.basename(img).replace('.jpg', '')
            chosen_df = small4k_df[small4k_df['image_id'] == int(image_id)]
            subject = deutsch_df[deutsch_df['Anforderungsnummer'] == int(image_id)]['Aufnahmenummer']
            chosen_df = chosen_df.assign(subject_id=subject.values.item(0))
            final_df_4K = final_df_4K.append(chosen_df[0:1])
            final_df_4K.to_csv(final_path_4K, sep=',', index=False)


    def cxr14_normalizer_resizer(self):
        base_path = '/mnt/hdd/Share/NIH_ChestX-ray14/CXR14/files'

        flag = 0
        final_df = pd.DataFrame(columns=['image_id', 'img_rel_path', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                                            'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis',	'pleural_thickening', 'hernia', 'no_finding',
                                         'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                                         'n_y_pixels', 'x_spacing', 'y_spacing'])

        label_path = '/mnt/hdd/Share/NIH_ChestX-ray14/cxr14_master_list.csv'
        final_df_output_path = '/mnt/hdd/Share/NIH_ChestX-ray14/final_cxr14_master_list.csv'
        df = pd.read_csv(label_path, sep=',')

        file_list = glob.glob(os.path.join(base_path, '*/*/*'))

        for file_path in tqdm(file_list):

            try:
                image = cv2.imread(file_path)

                # color to gray
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # resize
                resize_ratio = np.divide((HEIGHT, WIDTH), img.shape)
                img = zoom(img, resize_ratio, order=2)

                # normalization
                min_ = np.min(img)
                max_ = np.max(img)
                scale = max_ - min_
                img = (img - min_) / scale

                # converting to the range [0 255]
                img = img_as_ubyte(img)

                # histogram equalization
                img = cv2.equalizeHist(img)
                output_path = file_path.replace('/files/', '/preprocessed/')

                image_id = os.path.basename(output_path)
                basename2 = os.path.basename(os.path.dirname(output_path))
                basename3 = os.path.basename(os.path.dirname(os.path.dirname(output_path)))
                img_rel_path = os.path.join(basename3, basename2, image_id)

                chosen_df = df[df['image_id'] == image_id]

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                cv2.imwrite(output_path, img)

                tempp = pd.DataFrame(
                    [[chosen_df['image_id'].values[0], img_rel_path, chosen_df['patient_id'].values[0], chosen_df['split'].values[0],
                      chosen_df['atelectasis'].values[0], chosen_df['cardiomegaly'].values[0], chosen_df['effusion'].values[0],
                                            chosen_df['infiltration'].values[0], chosen_df['mass'].values[0], chosen_df['nodule'].values[0],
                      chosen_df['pneumonia'].values[0], chosen_df['pneumothorax'].values[0], chosen_df['consolidation'].values[0],
                     chosen_df['edema'].values[0], chosen_df['emphysema'].values[0], chosen_df['fibrosis'].values[0],
                      chosen_df['pleural_thickening'].values[0], chosen_df['hernia'].values[0], chosen_df['no_finding'].values[0],
                                         chosen_df['followup_num'].values[0], chosen_df['age'].values[0], chosen_df['gender'].values[0],
                      chosen_df['view_position'].values[0], chosen_df['n_x_pixels'].values[0],
                                         chosen_df['n_y_pixels'].values[0], chosen_df['x_spacing'].values[0], chosen_df['y_spacing'].values[0]]],
                    columns=['image_id', 'img_rel_path', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                                            'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis',	'pleural_thickening', 'hernia', 'no_finding',
                                         'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                                         'n_y_pixels', 'x_spacing', 'y_spacing'])
                final_df = final_df.append(tempp)
                final_df.to_csv(final_df_output_path, sep=',', index=False)

            except:
                flag += 1
                print(flag, file_path)

        final_df = final_df.sort_values(['split'])
        final_df.to_csv(final_df_output_path, sep=',', index=False)




class csv_summarizer():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"):
        pass


    def vindr(self):

        final_train = pd.DataFrame(columns=['image_id', 'split', 'Aortic enlargement', 'Atelectasis', 'Calcification',
                                            'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
                     'ILD', 'Infiltration',	'Lung Opacity',	'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
                                            'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                     'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia',
                                            'Tuberculosis', 'Other diseases', 'No finding'])

        trainlabel_path = '/home/soroosh/Documents/datasets/XRay/vindr-cxr1/preprocessed/image_labels_train.csv'
        trainlabel = pd.read_csv(trainlabel_path, sep=',')

        disease_list = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
                     'ILD', 'Infiltration',	'Lung Opacity',	'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                     'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']

        train_list = glob.glob('/home/soroosh/Documents/datasets/XRay/vindr-cxr1/preprocessed/train/*')
        train_list.sort()

        for image_path in tqdm(train_list):

            image_idlist = os.path.basename(image_path)
            image_idlist = image_idlist.split(".")
            assert len(image_idlist) == 2
            image_id = image_idlist[0]
            image = trainlabel[trainlabel['image_id'] == image_id]

            value_list =[]
            for disease in disease_list:
                if image[disease].mean() < 0.5:
                    value_list.append(int(0))
                else:
                    value_list.append(int(1))

            tempp = pd.DataFrame([[image_id, 'train', value_list[0], value_list[1], value_list[2], value_list[3], value_list[4], value_list[5], value_list[6],
                                   value_list[7], value_list[8], value_list[9], value_list[10], value_list[11], value_list[12], value_list[13],
                                   value_list[14], value_list[15], value_list[16], value_list[17], value_list[18], value_list[19], value_list[20],
                                   value_list[21], value_list[22], value_list[23], value_list[24], value_list[25], value_list[26], value_list[27]]],
                                 columns=['image_id', 'split', 'Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
                     'ILD', 'Infiltration',	'Lung Opacity',	'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                     'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding'])
            final_train = final_train.append(tempp)

        final_train.to_csv('/home/soroosh/Documents/datasets/XRay/vindr-cxr1/preprocessed/train_master_list.csv', sep=',', index=False)



    def UKA(self):
        final_df = pd.DataFrame(columns=['image_id', 'split', 'birth_date', 'examination_date', 'study_time',
                                            'patient_sex', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',	'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        label_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/onehot_UKA_master_list.csv'
        output_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/UKA_master_list.csv'
        df = pd.read_csv(label_path, sep=',')

        for index, row in tqdm(df.iterrows()):
            if row['cardiomegaly_1.0'] == 1:
                cardiomegaly = 1
            elif row['cardiomegaly_2.0'] == 1:
                cardiomegaly = 2
            elif row['cardiomegaly_3.0'] == 1:
                cardiomegaly = 3
            elif row['cardiomegaly_4.0'] == 1:
                cardiomegaly = 4
            elif row['cardiomegaly_5.0'] == 1:
                cardiomegaly = 5
            else:
                cardiomegaly = 0

            if row['congestion_1.0'] == 1:
                congestion = 1
            elif row['congestion_2.0'] == 1:
                congestion = 2
            elif row['congestion_3.0'] == 1:
                congestion = 3
            elif row['congestion_4.0'] == 1:
                congestion = 4
            elif row['congestion_5.0'] == 1:
                congestion = 5
            else:
                congestion = 0

            if row['pleural_effusion_right_1.0'] == 1:
                pleural_effusion_right = 1
            elif row['pleural_effusion_right_2.0'] == 1:
                pleural_effusion_right = 2
            elif row['pleural_effusion_right_3.0'] == 1:
                pleural_effusion_right = 3
            elif row['pleural_effusion_right_4.0'] == 1:
                pleural_effusion_right = 4
            elif row['pleural_effusion_right_5.0'] == 1:
                pleural_effusion_right = 5
            else:
                pleural_effusion_right = 0

            if row['pleural_effusion_left_1.0'] == 1:
                pleural_effusion_left = 1
            elif row['pleural_effusion_left_2.0'] == 1:
                pleural_effusion_left = 2
            elif row['pleural_effusion_left_3.0'] == 1:
                pleural_effusion_left = 3
            elif row['pleural_effusion_left_4.0'] == 1:
                pleural_effusion_left = 4
            elif row['pleural_effusion_left_5.0'] == 1:
                pleural_effusion_left = 5
            else:
                pleural_effusion_left = 0

            if row['pneumonic_infiltrates_right_1.0'] == 1:
                pneumonic_infiltrates_right = 1
            elif row['pneumonic_infiltrates_right_2.0'] == 1:
                pneumonic_infiltrates_right = 2
            elif row['pneumonic_infiltrates_right_3.0'] == 1:
                pneumonic_infiltrates_right = 3
            elif row['pneumonic_infiltrates_right_4.0'] == 1:
                pneumonic_infiltrates_right = 4
            elif row['pneumonic_infiltrates_right_5.0'] == 1:
                pneumonic_infiltrates_right = 5
            else:
                pneumonic_infiltrates_right = 0

            if row['pneumonic_infiltrates_left_1.0'] == 1:
                pneumonic_infiltrates_left = 1
            elif row['pneumonic_infiltrates_left_2.0'] == 1:
                pneumonic_infiltrates_left = 2
            elif row['pneumonic_infiltrates_left_3.0'] == 1:
                pneumonic_infiltrates_left = 3
            elif row['pneumonic_infiltrates_left_4.0'] == 1:
                pneumonic_infiltrates_left = 4
            elif row['pneumonic_infiltrates_left_5.0'] == 1:
                pneumonic_infiltrates_left = 5
            else:
                pneumonic_infiltrates_left = 0

            if row['Belstörungen_re_1.0'] == 1:
                disturbances_right = 1
            elif row['Belstörungen_re_2.0'] == 1:
                disturbances_right = 2
            elif row['Belstörungen_re_3.0'] == 1:
                disturbances_right = 3
            elif row['Belstörungen_re_4.0'] == 1:
                disturbances_right = 4
            elif row['Belstörungen_re_5.0'] == 1:
                disturbances_right = 5
            else:
                disturbances_right = 0

            if row['Belstörungen_li_1.0'] == 1:
                disturbances_left = 1
            elif row['Belstörungen_li_2.0'] == 1:
                disturbances_left = 2
            elif row['Belstörungen_li_3.0'] == 1:
                disturbances_left = 3
            elif row['Belstörungen_li_4.0'] == 1:
                disturbances_left = 4
            elif row['Belstörungen_li_5.0'] == 1:
                disturbances_left = 5
            else:
                disturbances_left = 0

            if row['Pneumothorax_re_1.0'] == 1:
                pneumothorax_right = 1
            elif row['Pneumothorax_re_2.0'] == 1:
                pneumothorax_right = 2
            elif row['Pneumothorax_re_3.0'] == 1:
                pneumothorax_right = 3
            elif row['Pneumothorax_re_4.0'] == 1:
                pneumothorax_right = 4
            elif row['Pneumothorax_re_5.0'] == 1:
                pneumothorax_right = 5
            elif row['Pneumothorax_re_6.0'] == 1:
                pneumothorax_right = 6
            elif row['Pneumothorax_re_7.0'] == 1:
                pneumothorax_right = 7
            else:
                pneumothorax_right = 0

            if row['Pneumothorax_li_1.0'] == 1:
                pneumothorax_left = 1
            elif row['Pneumothorax_li_2.0'] == 1:
                pneumothorax_left = 2
            elif row['Pneumothorax_li_3.0'] == 1:
                pneumothorax_left = 3
            elif row['Pneumothorax_li_4.0'] == 1:
                pneumothorax_left = 4
            elif row['Pneumothorax_li_5.0'] == 1:
                pneumothorax_left = 5
            elif row['Pneumothorax_li_6.0'] == 1:
                pneumothorax_left = 6
            elif row['Pneumothorax_li_7.0'] == 1:
                pneumothorax_left = 7
            else:
                pneumothorax_left = 0

            tempp = pd.DataFrame([[row['image_id'], row['split'], row['birth_date'], row['examination_date'], row['StudyTime'],
                                            row['PatientSex'], row['ExposureinuAs'], cardiomegaly, congestion, pleural_effusion_right, pleural_effusion_left,
                                   pneumonic_infiltrates_right, pneumonic_infiltrates_left, disturbances_right, disturbances_left, pneumothorax_right, pneumothorax_left, row['subject_id']]],
                                 columns=['image_id', 'split', 'birth_date', 'examination_date', 'study_time',
                                            'patient_sex', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left',	'disturbances_right',	'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])
            final_df = final_df.append(tempp)
            final_df.to_csv(output_path, sep=',', index=False)

        final_df.to_csv(output_path, sep=',', index=False)


    def cxr14(self):
        final_df = pd.DataFrame(columns=['image_id', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                                            'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis',	'pleural_thickening', 'hernia', 'no_finding',
                                         'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                                         'n_y_pixels', 'x_spacing', 'y_spacing'])

        label_path = '/mnt/hdd/Share/NIH_ChestX-ray14/Data_Entry_2017_v2020new.csv'
        split_path = '/mnt/hdd/Share/NIH_ChestX-ray14/test_list.txt'
        output_path = '/mnt/hdd/Share/NIH_ChestX-ray14/cxr14_master_list.csv'
        df = pd.read_csv(label_path, sep=',')
        split_df = pd.read_csv(split_path, sep=',')
        test_list = split_df['name'].to_list()

        for index, row in tqdm(df.iterrows()):
            if 'Cardiomegaly' in row['finding_labels']:
                cardiomegaly = 1
            else:
                cardiomegaly = 0

            if 'No Finding' in row['finding_labels']:
                no_finding = 1
            else:
                no_finding = 0

            if 'Infiltration' in row['finding_labels']:
                infiltration = 1
            else:
                infiltration = 0

            if 'Hernia' in row['finding_labels']:
                hernia = 1
            else:
                hernia = 0

            if 'Emphysema' in row['finding_labels']:
                emphysema = 1
            else:
                emphysema = 0

            if 'Effusion' in row['finding_labels']:
                effusion = 1
            else:
                effusion = 0

            if 'Atelectasis' in row['finding_labels']:
                atelectasis = 1
            else:
                atelectasis = 0

            if 'Pneumothorax' in row['finding_labels']:
                pneumothorax = 1
            else:
                pneumothorax = 0

            if 'Mass' in row['finding_labels']:
                mass = 1
            else:
                mass = 0

            if 'Nodule' in row['finding_labels']:
                nodule = 1
            else:
                nodule = 0

            if 'Pleural_Thickening' in row['finding_labels']:
                pleural_thickening = 1
            else:
                pleural_thickening = 0

            if 'Fibrosis' in row['finding_labels']:
                fibrosis = 1
            else:
                fibrosis = 0

            if 'Consolidation' in row['finding_labels']:
                consolidation = 1
            else:
                consolidation = 0

            if 'Edema' in row['finding_labels']:
                edema = 1
            else:
                edema = 0

            if 'Pneumonia' in row['finding_labels']:
                pneumonia = 1
            else:
                pneumonia = 0

            if row['image_id'] in test_list:
                split = 'test'
            else:
                split = 'train'

            tempp = pd.DataFrame([[row['image_id'], row['patient_id'], split, atelectasis, cardiomegaly, effusion,
                                            infiltration, mass, nodule, pneumonia, pneumothorax, consolidation,
                     edema, emphysema, fibrosis,	pleural_thickening, hernia, no_finding,
                                         row['followup_num'], row['age'], row['gender'], row['view_position'], row['n_x_pixels'],
                                         row['n_y_pixels'], row['x_spacing'], row['y_spacing']]],
                                 columns=['image_id', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                                            'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis',	'pleural_thickening', 'hernia', 'no_finding',
                                         'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                                         'n_y_pixels', 'x_spacing', 'y_spacing'])
            final_df = final_df.append(tempp)
            final_df.to_csv(output_path, sep=',', index=False)

        final_df = final_df.sort_values(['split'])
        final_df.to_csv(output_path, sep=',', index=False)




class csv_preprocess_chexpert():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"):
        pass


    def class_num_change(self):
        """
        Class 0 will stay 0: "negative"
        Class 1 will stay 1: "positive"
        Class -1 will become class 3: "uncertain positive"
        Class NaN will become class 2: not given; not mentioned in the report
        """

        input_path = "/home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/preprocessed/valid.csv"
        newoutput_path = "/home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/preprocessed/valid_master_list.csv"

        df1 = pd.read_csv(input_path, sep=',')

        df1[['no_finding', 'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion', 'edema',
             'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural_effusion', 'pleural_other',
             'fracture', 'support_devices']] = df1[
            ['no_finding', 'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion', 'edema',
             'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural_effusion', 'pleural_other',
             'fracture', 'support_devices']].fillna(5).astype(int)

        df1.loc[df1.atelectasis == -1, 'atelectasis'] = 3
        df1.loc[df1.cardiomegaly == -1, 'cardiomegaly'] = 3
        df1.loc[df1.consolidation == -1, 'consolidation'] = 3
        df1.loc[df1.edema == -1, 'edema'] = 3
        df1.loc[df1.enlarged_cardiomediastinum == -1, 'enlarged_cardiomediastinum'] = 3
        df1.loc[df1.fracture == -1, 'fracture'] = 3
        df1.loc[df1.lung_lesion == -1, 'lung_lesion'] = 3
        df1.loc[df1.lung_opacity == -1, 'lung_opacity'] = 3
        df1.loc[df1.no_finding == -1, 'no_finding'] = 3
        df1.loc[df1.pleural_effusion == -1, 'pleural_effusion'] = 3
        df1.loc[df1.pleural_other == -1, 'pleural_other'] = 3
        df1.loc[df1.pneumonia == -1, 'pneumonia'] = 3
        df1.loc[df1.pneumothorax == -1, 'pneumothorax'] = 3
        df1.loc[df1.support_devices == -1, 'support_devices'] = 3

        df1.loc[df1.atelectasis == 5, 'atelectasis'] = 2
        df1.loc[df1.cardiomegaly == 5, 'cardiomegaly'] = 2
        df1.loc[df1.consolidation == 5, 'consolidation'] = 2
        df1.loc[df1.edema == 5, 'edema'] = 2
        df1.loc[df1.enlarged_cardiomediastinum == 5, 'enlarged_cardiomediastinum'] = 2
        df1.loc[df1.fracture == 5, 'fracture'] = 2
        df1.loc[df1.lung_lesion == 5, 'lung_lesion'] = 2
        df1.loc[df1.lung_opacity == 5, 'lung_opacity'] = 2
        df1.loc[df1.no_finding == 5, 'no_finding'] = 2
        df1.loc[df1.pleural_effusion == 5, 'pleural_effusion'] = 2
        df1.loc[df1.pleural_other == 5, 'pleural_other'] = 2
        df1.loc[df1.pneumonia == 5, 'pneumonia'] = 2
        df1.loc[df1.pneumothorax == 5, 'pneumothorax'] = 2
        df1.loc[df1.support_devices == 5, 'support_devices'] = 2

        df1.to_csv(newoutput_path, sep=',', index=False)



    def threetwo_remover(self):
        path = "/home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/master_list.csv"
        newoutput_path = "/home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/nothree_master_list.csv"

        final_data = pd.DataFrame(columns=['jpg_rel_path','split', 'gender', 'age', 'view', 'AP_PA', 'no_finding',
                             'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion',
                             'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax',
                             'pleural_effusion', 'pleural_other', 'fracture', 'support_devices'])

        df = pd.read_csv(path, sep=',')
        for index, row in tqdm(df.iterrows()):
            if (row['cardiomegaly'] < 3) and (row['lung_opacity'] < 3) and (row['lung_lesion'] < 3) and (row['pneumonia'] < 3) and (row['edema'] < 3):
            # if (row['cardiomegaly'] < 2):
                final_data = final_data.append(row)

        final_data.to_csv(newoutput_path, sep=',', index=False)



class csv_reducer():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"):
        pass


    def vindr(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['image_id', 'split', 'Aortic enlargement', 'Atelectasis', 'Calcification',
                                            'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
                     'ILD', 'Infiltration',	'Lung Opacity',	'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
                                            'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                     'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia',
                                            'Tuberculosis', 'Other diseases', 'No finding'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/vindr-cxr1/officialsoroosh_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/vindr-cxr1/new2000_officialsoroosh_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['split'] == 'train']
        valid_df = org_df[org_df['split'] == 'valid']
        test_df = org_df[org_df['split'] == 'test']

        train_list = train_df['image_id'].unique().tolist()
        random.shuffle(train_list)

        chosen_list = train_list[:num_images]

        for patient in tqdm(chosen_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['image_id'])
        final_df = final_df.append(valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def vindr_pediatric(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['image_id', 'split', 'No finding', 'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis',
                                        'Situs inversus', 'Pneumonia', 'Pleuro-pneumonia', 'Diagphramatic hernia', 'Tuberculosis', 'Congenital emphysema',
                                        'CPAM', 'Hyaline membrane disease', 'Mediastinal tumor', 'Lung tumor', 'rad_ID'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/vindr-pcxr/master_list_vindr-pcxr.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/vindr-pcxr/reduced_master_list_vindr-pcxr.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['split'] == 'train']
        # valid_df = org_df[org_df['split'] == 'valid']
        test_df = org_df[org_df['split'] == 'test']

        train_list = train_df['image_id'].unique().tolist()
        random.shuffle(train_list)

        chosen_list = train_list[:num_images]

        for patient in tqdm(chosen_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['image_id'])
        # final_df = final_df.append(valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def chexpert(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['jpg_rel_path','split', 'gender', 'age', 'view', 'AP_PA', 'no_finding',
                             'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion',
                             'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax',
                             'pleural_effusion', 'pleural_other', 'fracture', 'support_devices'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/nothree_master_list_20percenttest.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/CheXpert-v1.0/5000_nothree_master_list_20percenttest.csv'

        org_df = pd.read_csv(org_df_path, sep=',')
        org_df = org_df[org_df['view'] == 'Frontal']

        train_df = org_df[org_df['split'] == 'train']
        valid_df = org_df[org_df['split'] == 'valid']
        test_df = org_df[org_df['split'] == 'test']

        train_list = train_df['jpg_rel_path'].unique().tolist()
        random.shuffle(train_list)

        chosen_list = train_list[:num_images]

        for patient in tqdm(chosen_list):
            selected_patient_df = train_df[train_df['jpg_rel_path'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['jpg_rel_path'])
        final_df = final_df.append(valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def mimic(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['jpg_rel_path', 'report_rel_path', 'subject_id', 'study_id', 'split', 'view', 'available_views',
                     'n_x_pixels', 'n_y_pixels', 'atelectasis', 'cardiomegaly', 'consolidation', 'edema',
                     'enlarged_cardiomediastinum', 'fracture', 'lung_lesion', 'lung_opacity', 'no_finding',
                     'pleural_effusion', 'pleural_other', 'pneumonia', 'pneumothorax', 'support_devices', 'subset'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/MIMIC/nothree_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/MIMIC/reduced_nothree_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')
        PAview = org_df[org_df['view'] == 'PA']
        APview = org_df[org_df['view'] == 'AP']
        org_df = PAview.append(APview)

        train_df = org_df[org_df['split'] == 'train']
        valid_df = org_df[org_df['split'] == 'valid']
        test_df = org_df[org_df['split'] == 'test']

        train_list = train_df['jpg_rel_path'].unique().tolist()
        random.shuffle(train_list)

        chosen_list = train_list[:num_images]

        for patient in tqdm(chosen_list):
            selected_patient_df = train_df[train_df['jpg_rel_path'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['jpg_rel_path'])
        final_df = final_df.append(valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def cxr14(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['image_id', 'img_rel_path', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                                            'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis',	'pleural_thickening', 'hernia', 'no_finding',
                                         'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                                         'n_y_pixels', 'x_spacing', 'y_spacing'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/NIH_ChestX-ray14/officialsoroosh_cxr14_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/NIH_ChestX-ray14/reduced_officialsoroosh_cxr14_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['split'] == 'train']
        valid_df = org_df[org_df['split'] == 'valid']
        test_df = org_df[org_df['split'] == 'test']

        train_list = train_df['image_id'].unique().tolist()
        random.shuffle(train_list)

        chosen_list = train_list[:num_images]

        for patient in tqdm(chosen_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['image_id'])
        final_df = final_df.append(valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def UKA(self, num_images):

        # initiating the df
        # final_df = pd.DataFrame(columns=['image_id', 'split', 'subset', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
        #              'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',	'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])
        final_df = pd.DataFrame(columns=['image_id', 'split', 'subset', 'birth_date', 'examination_date', 'study_time',
                                            'patient_sex', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',	'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/DP_project_also_original/original_UKA_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/DP_project_also_original/15000_final_multitask_UKA_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['split'] == 'train']
        valid_df = org_df[org_df['split'] == 'valid']
        test_df = org_df[org_df['split'] == 'test']

        train_list = train_df['image_id'].unique().tolist()
        random.shuffle(train_list)

        chosen_list = train_list[:num_images]

        for patient in tqdm(chosen_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['image_id'])
        final_df = final_df.append(valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def UKA_test_reducer(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['image_id', 'split', 'subset', 'ExposureinuAs', 'cardiomegaly', 'congestion', 'pleural_effusion_right', 'pleural_effusion_left',
                     'pneumonic_infiltrates_right', 'pneumonic_infiltrates_left', 'disturbances_right',	'disturbances_left', 'pneumothorax_right', 'pneumothorax_left', 'subject_id'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/final_UKA_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/UKA/chest_radiograph/new_final_UKA_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['split'] == 'train']
        valid_df = org_df[org_df['split'] == 'valid']
        test_df = org_df[org_df['split'] == 'test']

        test_list = test_df['image_id'].unique().tolist()
        random.shuffle(test_list)

        chosen_list = test_list[:num_images]
        rest_list = test_list[num_images:]

        for patient in tqdm(chosen_list):
            selected_patient_df = test_df[test_df['image_id'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['image_id'])
        final_df = final_df.append(train_df)
        final_df = final_df.append(valid_df)

        for patient in tqdm(rest_list):
            selected_patient_df = test_df[test_df['image_id'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df.to_csv(output_df_path, sep=',', index=False)


    def coronahack(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['X_ray_image_name', 'Label', 'Dataset_type', 'Label_2_Virus_category', 'Label_1_Virus_category'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/Coronahack_Chest_XRay/officialsoroosh_coronahack_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/Coronahack_Chest_XRay/reduced_officialsoroosh_coronahack_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['Dataset_type'] == 'TRAIN']
        valid_df = org_df[org_df['Dataset_type'] == 'VALID']
        test_df = org_df[org_df['Dataset_type'] == 'TEST']

        train_list = train_df['X_ray_image_name'].unique().tolist()
        random.shuffle(train_list)

        chosen_list = train_list[:num_images]

        for patient in tqdm(chosen_list):
            selected_patient_df = train_df[train_df['X_ray_image_name'] == patient]
            final_df = final_df.append(selected_patient_df)

        final_df = final_df.sort_values(['X_ray_image_name'])
        final_df = final_df.append(valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def vindr_validmaker(self, num_images):

        # initiating the df
        final_valid_df = pd.DataFrame(columns=['image_id', 'split', 'Aortic enlargement', 'Atelectasis', 'Calcification',
                                            'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
                     'ILD', 'Infiltration',	'Lung Opacity',	'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
                                            'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                     'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia',
                                            'Tuberculosis', 'Other diseases', 'No finding'])
        final_train_df = pd.DataFrame(columns=['image_id', 'split', 'Aortic enlargement', 'Atelectasis', 'Calcification',
                                            'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
                     'ILD', 'Infiltration',	'Lung Opacity',	'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
                                            'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                     'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia',
                                            'Tuberculosis', 'Other diseases', 'No finding'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/vindr-cxr1/master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/vindr-cxr1/officialsoroosh_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['split'] == 'train']
        test_df = org_df[org_df['split'] == 'test']

        train_files = train_df['image_id'].unique().tolist()
        random.shuffle(train_files)

        valid_list = train_files[:num_images]
        train_list = train_files[num_images:]

        for patient in tqdm(valid_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_valid_df = final_valid_df.append(selected_patient_df)

        for patient in tqdm(train_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_train_df = final_train_df.append(selected_patient_df)

        final_valid_df.loc[final_valid_df.split == 'train', 'split'] = 'valid'
        final_valid_df = final_valid_df.sort_values(['image_id'])
        final_train_df = final_train_df.sort_values(['image_id'])


        final_df = final_train_df.append(final_valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def coronahack_validmaker(self, num_images):

        # initiating the df
        final_valid_df = pd.DataFrame(columns=['X_ray_image_name', 'Label', 'Dataset_type', 'Label_2_Virus_category', 'Label_1_Virus_category'])
        final_train_df = pd.DataFrame(columns=['X_ray_image_name', 'Label', 'Dataset_type', 'Label_2_Virus_category', 'Label_1_Virus_category'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/Coronahack_Chest_XRay/coronahack_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/Coronahack_Chest_XRay/officialsoroosh_coronahack_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['Dataset_type'] == 'TRAIN']
        test_df = org_df[org_df['Dataset_type'] == 'TEST']

        train_files = train_df['X_ray_image_name'].unique().tolist()
        random.shuffle(train_files)

        valid_list = train_files[:num_images]
        train_list = train_files[num_images:]

        for patient in tqdm(valid_list):
            selected_patient_df = train_df[train_df['X_ray_image_name'] == patient]
            final_valid_df = final_valid_df.append(selected_patient_df)

        for patient in tqdm(train_list):
            selected_patient_df = train_df[train_df['X_ray_image_name'] == patient]
            final_train_df = final_train_df.append(selected_patient_df)

        final_valid_df.loc[final_valid_df.Dataset_type == 'TRAIN', 'Dataset_type'] = 'VALID'
        final_valid_df = final_valid_df.sort_values(['X_ray_image_name'])
        final_train_df = final_train_df.sort_values(['X_ray_image_name'])

        final_df = final_train_df.append(final_valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)


    def cxr14_validmaker(self, num_images):

        final_train_df = pd.DataFrame(columns=['image_id', 'img_rel_path', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                                            'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis',	'pleural_thickening', 'hernia', 'no_finding',
                                         'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                                         'n_y_pixels', 'x_spacing', 'y_spacing'])
        final_valid_df = pd.DataFrame(columns=['image_id', 'img_rel_path', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                                            'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis',	'pleural_thickening', 'hernia', 'no_finding',
                                         'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                                         'n_y_pixels', 'x_spacing', 'y_spacing'])

        org_df_path = '/home/soroosh/Documents/datasets/XRay/NIH_ChestX-ray14/final_cxr14_master_list.csv'
        output_df_path = '/home/soroosh/Documents/datasets/XRay/NIH_ChestX-ray14/officialsoroosh_cxr14_master_list.csv'

        org_df = pd.read_csv(org_df_path, sep=',')

        train_df = org_df[org_df['split'] == 'train']
        test_df = org_df[org_df['split'] == 'test']

        train_files = train_df['image_id'].unique().tolist()
        random.shuffle(train_files)

        valid_list = train_files[:num_images]
        train_list = train_files[num_images:]

        for patient in tqdm(valid_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_valid_df = final_valid_df.append(selected_patient_df)

        for patient in tqdm(train_list):
            selected_patient_df = train_df[train_df['image_id'] == patient]
            final_train_df = final_train_df.append(selected_patient_df)

        final_valid_df.loc[final_valid_df.split == 'train', 'split'] = 'valid'
        final_valid_df = final_valid_df.sort_values(['image_id'])
        final_train_df = final_train_df.sort_values(['image_id'])


        final_df = final_train_df.append(final_valid_df)
        final_df = final_df.append(test_df)
        final_df.to_csv(output_df_path, sep=',', index=False)





if __name__ == '__main__':
    # handler = csv_preprocess_mimic()
    # handler.csv_creator()
    # handler.class_num_change()
    # handler.threetwo_remover()
    # hendler3 = csv_summarizer()
    # hendler3.vindr()
    # hendler3.cxr14()

    # handler2 = normalizer_resizer()
    # handler2.mimic_normalizer_resizer()
    # handler2.vindr_normalizer_resizer()
    # handler2.chexpert_normalizer_resizer()
    # handler2.pediatric_corona_normalizer_resizer()
    # handler2.UKA_normalizer_resizer()
    # handler2.cxr14_normalizer_resizer()

    # handler4 = csv_preprocess_chexpert()
    # handler4.class_num_change()
    # handler4.threetwo_remover()

    handler5 = csv_reducer()
    # handler5.coronahack(num_images=2000)
    # handler5.mimic(num_images=2000)
    # handler5.cxr14_validmaker(num_images=3000)
    handler5.vindr_pediatric(num_images=5000)
    # handler5.UKA_test_reducer(num_images=4000)
    # handler5.chexpert(num_images=2000)
    # handler5.UKA(num_images=15000)
