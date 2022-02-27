"""
Created on Feb 1, 2022.
main_2D_chestx.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from models.Xception_model import Xception
from models.resnet18 import ResNet18
from Train_Valid_chestx import Training, load_pretrained_model
from Prediction_chestx import Prediction
from data.data_provider import data_loader
from data.data_handler_pv_defect import ChallengeDataset

import warnings
warnings.filterwarnings('ignore')
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]



def main_train_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', chosen_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  subsets=['p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19']):
    """Main function for multi label training + validation for directly 2d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.

        chosen_labels: list of integers
            index of the classes that we want to have in our training.

        subsets: list of strings
            name of the data subsets from MIMIC dataset that we want to have in our training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    model_info = params['Network']
    model_info['subsets'] = subsets
    params['Network'] = model_info
    write_config(params, cfg_path, sort_keys=True)

    # Changeable network parameters
    model = Xception(num_classes=len(chosen_labels))
    # model = ResNet18(n_out_classes=len(chosen_labels))
    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    train_dataset = data_loader(cfg_path=cfg_path, mode='train', chosen_labels=chosen_labels, subsets=subsets)

    # class weights corresponding to the dataset
    pos_weight = train_dataset.pos_weight(chosen_labels=chosen_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=40)
    if valid:
        valid_dataset = data_loader(cfg_path=cfg_path, mode='valid', chosen_labels=chosen_labels, subsets=subsets)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=True, drop_last=True, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, chosen_labels=chosen_labels)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=pos_weight)
    trainer.train_epoch(train_loader=train_loader, batch_size=params['Network']['batch_size'], valid_loader=valid_loader)




def main_test_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", experiment_name='name'):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    label_names = params['label_names']
    chosen_labels = params['Network']['chosen_labels']
    subsets = params['Network']['subsets']

    # Changeable network parameters
    model = Xception(num_classes=len(chosen_labels))
    # model = ResNet18(n_out_classes=len(chosen_labels))

    test_dataset = data_loader(cfg_path=cfg_path, mode='test', chosen_labels=chosen_labels, subsets=subsets)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=False, num_workers=40)

    # Initialize prediction
    predictor = Prediction(cfg_path, chosen_labels)
    predictor.setup_model(model=model)
    accuracy_disease, sensitivity_disease, specifity_disease, F1_disease = predictor.evaluate_2D(test_loader, params['Network']['batch_size'])

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\tTotal accuracy: {accuracy_disease.mean() * 100:.2f}% | total sensitivity: {sensitivity_disease.mean() * 100:.2f}%'
          f' | total specifity: {specifity_disease.mean() * 100:.2f}% | total F1 score: {F1_disease.mean() * 100:.2f}%')
    print('\nIndividual accuracy scores:')
    for idx, pathology in enumerate(chosen_labels):
        print(f'\t{label_names[pathology]}: {accuracy_disease[idx] * 100:.2f}%')

    print('\nIndividual sensitivity scores:')
    for idx, pathology in enumerate(chosen_labels):
        print(f'\t{label_names[pathology]}: {sensitivity_disease[idx] * 100:.2f}%')

    print('\nIndividual specifity scores:')
    for idx, pathology in enumerate(chosen_labels):
        print(f'\t{label_names[pathology]}: {specifity_disease[idx] * 100:.2f}%')

    print('\nIndividual F1 scores:')
    for idx, pathology in enumerate(chosen_labels):
        print(f'\t{label_names[pathology]}: {F1_disease[idx] * 100:.2f}%')

    print('------------------------------------------------------'
          '----------------------------------')

    # # saving the stats
    # mesg = f'\n\n----------------------------------------------------------------------------------------\n' \
    #        f'\tTotal Accuracy: {accuracy_disease.mean() * 100:.2f}% | Total sensitivity: {sensitivity_disease.mean() * 100:.2f}%' \
    #        f' | Total specifity: {specifity_disease.mean() * 100:.2f}%' \
    #        f'\n\nIndividual Accuracy scores:' \
    #        f'\tAtelectasis: {accuracy_disease[0] * 100:.2f}% | Cardiomegaly: {accuracy_disease[1] * 100:.2f}% ' \
    #       f'| Consolidation: {accuracy_disease[2] * 100:.2f}% | Edema: {accuracy_disease[3] * 100:.2f}%' \
    #        f'\tEnlarged Cardiomediastinum: {accuracy_disease[4] * 100:.2f}% | Fracture: {accuracy_disease[5] * 100:.2f}% ' \
    #        f'| Lung Lesion: {accuracy_disease[6] * 100:.2f}% | Lung Opacity: {accuracy_disease[7] * 100:.2f}%' \
    #     f'\tNo Finding: {accuracy_disease[8] * 100:.2f}% | Pleural Effusion: {accuracy_disease[9] * 100:.2f}% ' \
    #        f'| Pleural Other: {accuracy_disease[10] * 100:.2f}% | Pneumonia: {accuracy_disease[11] * 100:.2f}%' \
    #        f'\tPneumothorax: {accuracy_disease[12] * 100:.2f}% | Support Devices: {accuracy_disease[13] * 100:.2f}%' \
    #        f'\n\nIndividual sensitivity scores:' \
    #        f'\tAtelectasis: {sensitivity_disease[0] * 100:.2f}% | Cardiomegaly: {sensitivity_disease[1] * 100:.2f}% ' \
    #       f'| Consolidation: {sensitivity_disease[2] * 100:.2f}% | Edema: {sensitivity_disease[3] * 100:.2f}%' \
    #        f'\tEnlarged Cardiomediastinum: {sensitivity_disease[4] * 100:.2f}% | Fracture: {sensitivity_disease[5] * 100:.2f}% ' \
    #        f'| Lung Lesion: {sensitivity_disease[6] * 100:.2f}% | Lung Opacity: {sensitivity_disease[7] * 100:.2f}%' \
    #     f'\tNo Finding: {sensitivity_disease[8] * 100:.2f}% | Pleural Effusion: {sensitivity_disease[9] * 100:.2f}% ' \
    #        f'| Pleural Other: {sensitivity_disease[10] * 100:.2f}% | Pneumonia: {sensitivity_disease[11] * 100:.2f}%' \
    #        f'\tPneumothorax: {sensitivity_disease[12] * 100:.2f}% | Support Devices: {sensitivity_disease[13] * 100:.2f}%' \
    #        f'\n\nIndividual specifity scores:' \
    #        f'\tAtelectasis: {specifity_disease[0] * 100:.2f}% | Cardiomegaly: {specifity_disease[1] * 100:.2f}% ' \
    #       f'| Consolidation: {specifity_disease[2] * 100:.2f}% | Edema: {specifity_disease[3] * 100:.2f}%' \
    #        f'\tEnlarged Cardiomediastinum: {specifity_disease[4] * 100:.2f}% | Fracture: {specifity_disease[5] * 100:.2f}% ' \
    #        f'| Lung Lesion: {specifity_disease[6] * 100:.2f}% | Lung Opacity: {specifity_disease[7] * 100:.2f}%' \
    #     f'\tNo Finding: {specifity_disease[8] * 100:.2f}% | Pleural Effusion: {specifity_disease[9] * 100:.2f}% ' \
    #        f'| Pleural Other: {specifity_disease[10] * 100:.2f}% | Pneumonia: {specifity_disease[11] * 100:.2f}%' \
    #        f'\tPneumothorax: {specifity_disease[12] * 100:.2f}% | Support Devices: {specifity_disease[13] * 100:.2f}%'
    # with open(os.path.join(params['target_dir'], params['stat_log_path'], '/test_results'), 'a') as f:
    #     f.write(mesg)





def main_train_solar_cells_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', chosen_labels=[0, 1]):
    """Main function for multi label training + validation for directly 2d-wise

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.

        chosen_labels: list of integers
            index of the classes that we want to have in our training.

        subsets: list of strings
            name of the data subsets from MIMIC dataset that we want to have in our training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = Xception(num_classes=len(chosen_labels))
    # model = ResNet18(n_out_classes=len(chosen_labels))
    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                transforms.Normalize(train_mean, train_std)])
    train_dataset =  ChallengeDataset(cfg_path=cfg_path, transform=trans, chosen_labels=chosen_labels, training=True)
    # class weights corresponding to the dataset
    pos_weight = train_dataset.pos_weight(chosen_labels=chosen_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=40)

    if valid:
        trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                    transforms.Normalize(train_mean, train_std)])
        valid_dataset = ChallengeDataset(cfg_path=cfg_path, transform=trans,
                                         chosen_labels=chosen_labels, training=False)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=True, drop_last=True, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, chosen_labels=chosen_labels)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=pos_weight)
    trainer.train_epoch(train_loader=train_loader, batch_size=params['Network']['batch_size'], valid_loader=valid_loader)





if __name__ == '__main__':
    # delete_experiment(experiment_name='xception_solar_2e5', global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml")
    main_train_solar_cells_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml",
                  valid=True, resume=False, augment=False, experiment_name='xception_solar_2e5',
                  chosen_labels=[1, 2])
    # main_train_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml",
    #               valid=True, resume=False, augment=False, experiment_name='xception_p10-11_weight_3_labels_2e5',
    #               chosen_labels=[0, 1, 7], subsets=['p10', 'p11'])
    # main_test_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", experiment_name='first_try')
