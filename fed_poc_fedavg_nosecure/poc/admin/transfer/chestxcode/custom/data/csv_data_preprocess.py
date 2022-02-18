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
        Class 1 will stay 1: "positive"
        Class 0 will become class 2: "negative"
        Class -1 will become class 3: "not sure"
        Class NaN will become class 0: background (not given; none of the "positive", "negative", "not sure")
        """

        output_path = "/home/soroosh/Documents/datasets/MIMIC/mimic_master_list.csv"
        newoutput_path = "/home/soroosh/Documents/datasets/MIMIC/newmimic_master_list.csv"

        df1 = pd.read_csv(output_path, sep=',')

        df1[['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture',
             'lung_lesion', 'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
             'pneumothorax', 'support_devices']] = df1[
            ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged_cardiomediastinum', 'fracture',
             'lung_lesion', 'lung_opacity', 'no_finding', 'pleural_effusion', 'pleural_other', 'pneumonia',
             'pneumothorax', 'support_devices']].fillna(5).astype(int)

        df1.loc[df1.atelectasis == 0, 'atelectasis'] = 2
        df1.loc[df1.cardiomegaly == 0, 'cardiomegaly'] = 2
        df1.loc[df1.consolidation == 0, 'consolidation'] = 2
        df1.loc[df1.edema == 0, 'edema'] = 2
        df1.loc[df1.enlarged_cardiomediastinum == 0, 'enlarged_cardiomediastinum'] = 2
        df1.loc[df1.fracture == 0, 'fracture'] = 2
        df1.loc[df1.lung_lesion == 0, 'lung_lesion'] = 2
        df1.loc[df1.lung_opacity == 0, 'lung_opacity'] = 2
        df1.loc[df1.no_finding == 0, 'no_finding'] = 2
        df1.loc[df1.pleural_effusion == 0, 'pleural_effusion'] = 2
        df1.loc[df1.pleural_other == 0, 'pleural_other'] = 2
        df1.loc[df1.pneumonia == 0, 'pneumonia'] = 2
        df1.loc[df1.pneumothorax == 0, 'pneumothorax'] = 2
        df1.loc[df1.support_devices == 0, 'support_devices'] = 2

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

        df1.loc[df1.atelectasis == 5, 'atelectasis'] = 0
        df1.loc[df1.cardiomegaly == 5, 'cardiomegaly'] = 0
        df1.loc[df1.consolidation == 5, 'consolidation'] = 0
        df1.loc[df1.edema == 5, 'edema'] = 0
        df1.loc[df1.enlarged_cardiomediastinum == 5, 'enlarged_cardiomediastinum'] = 0
        df1.loc[df1.fracture == 5, 'fracture'] = 0
        df1.loc[df1.lung_lesion == 5, 'lung_lesion'] = 0
        df1.loc[df1.lung_opacity == 5, 'lung_opacity'] = 0
        df1.loc[df1.no_finding == 5, 'no_finding'] = 0
        df1.loc[df1.pleural_effusion == 5, 'pleural_effusion'] = 0
        df1.loc[df1.pleural_other == 5, 'pleural_other'] = 0
        df1.loc[df1.pneumonia == 5, 'pneumonia'] = 0
        df1.loc[df1.pneumothorax == 5, 'pneumothorax'] = 0
        df1.loc[df1.support_devices == 5, 'support_devices'] = 0

        df1.to_csv(newoutput_path, sep=',', index=False)




if __name__ == '__main__':
    handler = csv_preprocess_mimic()
    handler.csv_creator()