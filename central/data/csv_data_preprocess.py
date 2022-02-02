"""
Created on Feb 2, 2022.
csv_data_preprocess.py

creating a master list for mimic dataset.

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import os
import pdb
import pandas as pd
from tqdm import tqdm

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')




class csv_preprocess_mimic():
    def __init__(self, cfg_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml"):
        self.params = read_config(cfg_path)


    def csv_creator(self):
        """csv_creator"""

        output_path = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list.csv"

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
                                                                    row['dicom_id']]['path'].values[0]).replace('.dcm', '.jpg'))
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
            if index % 100 == 0:
                final_data = final_data.sort_values(['jpg_rel_path'])
                final_data.to_csv(output_path, sep=',', index=False)


        # sort based on name
        final_data = final_data.sort_values(['jpg_rel_path'])
        final_data.to_csv(output_path, sep=',', index=False)






if __name__ == '__main__':
    handler = csv_preprocess_mimic()
    handler.csv_creator()