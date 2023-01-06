"""
Created on Feb 2, 2022.
csv_data_preprocess.py

data preprocessing for X-Ray images.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""
import glob
import os
import pandas as pd
from tqdm import tqdm
import pydicom as dicom
import numpy as np
import cv2
from skimage.util import img_as_ubyte
from scipy.ndimage.interpolation import zoom
import random

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')

HEIGHT, WIDTH = 512, 512




class normalizer_resizer():
    def __init__(self, cfg_path="chestx/config/config.yaml"):
        pass

    def mimic_normalizer_resizer(self):
        base_path = 'datasets/MIMIC'

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

        path = "datasets/vindr-cxr1/original"

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
        base_path = 'Chexpert_dataset/'
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

    def cxr14_normalizer_resizer(self):
        base_path = 'NIH_ChestX-ray14/CXR14/files'

        flag = 0
        final_df = pd.DataFrame(
            columns=['image_id', 'img_rel_path', 'patient_id', 'split', 'atelectasis', 'cardiomegaly', 'effusion',
                     'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                     'edema', 'emphysema', 'fibrosis', 'pleural_thickening', 'hernia', 'no_finding',
                     'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                     'n_y_pixels', 'x_spacing', 'y_spacing'])

        label_path = '/NIH_ChestX-ray14/cxr14_master_list.csv'
        final_df_output_path = '/NIH_ChestX-ray14/final_cxr14_master_list.csv'
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
                    [[chosen_df['image_id'].values[0], img_rel_path, chosen_df['patient_id'].values[0],
                      chosen_df['split'].values[0],
                      chosen_df['atelectasis'].values[0], chosen_df['cardiomegaly'].values[0],
                      chosen_df['effusion'].values[0],
                      chosen_df['infiltration'].values[0], chosen_df['mass'].values[0], chosen_df['nodule'].values[0],
                      chosen_df['pneumonia'].values[0], chosen_df['pneumothorax'].values[0],
                      chosen_df['consolidation'].values[0],
                      chosen_df['edema'].values[0], chosen_df['emphysema'].values[0], chosen_df['fibrosis'].values[0],
                      chosen_df['pleural_thickening'].values[0], chosen_df['hernia'].values[0],
                      chosen_df['no_finding'].values[0],
                      chosen_df['followup_num'].values[0], chosen_df['age'].values[0], chosen_df['gender'].values[0],
                      chosen_df['view_position'].values[0], chosen_df['n_x_pixels'].values[0],
                      chosen_df['n_y_pixels'].values[0], chosen_df['x_spacing'].values[0],
                      chosen_df['y_spacing'].values[0]]],
                    columns=['image_id', 'img_rel_path', 'patient_id', 'split', 'atelectasis', 'cardiomegaly',
                             'effusion',
                             'infiltration', 'mass', 'nodule', 'pneumonia', 'pneumothorax', 'consolidation',
                             'edema', 'emphysema', 'fibrosis', 'pleural_thickening', 'hernia', 'no_finding',
                             'followup_num', 'age', 'gender', 'view_position', 'n_x_pixels',
                             'n_y_pixels', 'x_spacing', 'y_spacing'])
                final_df = final_df.append(tempp)
                final_df.to_csv(final_df_output_path, sep=',', index=False)

            except:
                flag += 1
                print(flag, file_path)

        final_df = final_df.sort_values(['split'])
        final_df.to_csv(final_df_output_path, sep=',', index=False)



class csv_preprocess_chexpert():
    def __init__(self, cfg_path="chestx/config/config.yaml"):
        pass


    def class_num_change(self):
        """
        Class 0 will stay 0: "negative"
        Class 1 will stay 1: "positive"
        Class -1 will become class 3: "uncertain positive"
        Class NaN will become class 2: not given; not mentioned in the report
        """

        input_path = "/CheXpert-v1.0/preprocessed/valid.csv"
        newoutput_path = "/CheXpert-v1.0/preprocessed/valid_master_list.csv"

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
        path = "/CheXpert-v1.0/master_list.csv"
        newoutput_path = "/CheXpert-v1.0/nothree_master_list.csv"

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
    def __init__(self, cfg_path="chestx/config/config.yaml"):
        pass


    def vindr(self, num_images):

        # initiating the df
        final_df = pd.DataFrame(columns=['image_id', 'split', 'Aortic enlargement', 'Atelectasis', 'Calcification',
                                            'Cardiomegaly', 'Clavicle fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged PA',
                     'ILD', 'Infiltration',	'Lung Opacity',	'Lung cavity', 'Lung cyst', 'Mediastinal shift', 'Nodule/Mass',
                                            'Pleural effusion', 'Pleural thickening', 'Pneumothorax',
                     'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Pneumonia',
                                            'Tuberculosis', 'Other diseases', 'No finding'])

        org_df_path = '/xxxxx.csv'
        output_df_path = '/xxxx.csv'

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





if __name__ == '__main__':
    handler = normalizer_resizer()
    handler.cxr14_normalizer_resizer()