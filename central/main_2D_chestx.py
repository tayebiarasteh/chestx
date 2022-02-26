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

from config.serde import open_experiment, create_experiment, delete_experiment
from models.Xception_model import Xception
from models.resnet18 import ResNet18
from Train_Valid_chestx import Training
from Prediction_chestx import Prediction
from data.data_provider import data_loader

import warnings
warnings.filterwarnings('ignore')



def main_train_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name'):
    """Main function for training + validation for directly 2d-wise

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
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = Xception(num_classes=14)
    # model = ResNet18(n_out_classes=14)
    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    train_dataset = data_loader(cfg_path=cfg_path, mode='train')

    # class weights corresponding to the dataset
    pos_weight = train_dataset.pos_weight()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=False, drop_last=True, shuffle=True, num_workers=4)
    if valid:
        valid_dataset = data_loader(cfg_path=cfg_path, mode='valid')
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=False, drop_last=True, shuffle=False, num_workers=1)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=pos_weight)
    trainer.train_epoch(train_loader=train_loader, batch_size=params['Network']['batch_size'], valid_loader=valid_loader)




def main_test_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", experiment_name='name'):
    """Main function for prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = create_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    # Changeable network parameters
    model = Xception(num_classes=14)
    # model = ResNet18(n_out_classes=14)

    test_dataset = data_loader(cfg_path=cfg_path, mode='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=False, drop_last=True, shuffle=False, num_workers=4)

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)
    accuracy_disease, sensitivity_disease, specifity_disease = predictor.evaluate_2D(test_loader, params['Network']['batch_size'])

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\tTotal Accuracy: {accuracy_disease.mean() * 100:.2f}% | Total sensitivity: {sensitivity_disease.mean() * 100:.2f}%'
          f' | Total specifity: {specifity_disease.mean() * 100:.2f}%')
    print('\nIndividual Accuracy scores:')
    print(f'\tAtelectasis: {accuracy_disease[0] * 100:.2f}% | Cardiomegaly: {accuracy_disease[1] * 100:.2f}% '
          f'| Consolidation: {accuracy_disease[2] * 100:.2f}% | Edema: {accuracy_disease[3] * 100:.2f}%')
    print(f'\tEnlarged Cardiomediastinum: {accuracy_disease[4] * 100:.2f}% | Fracture: {accuracy_disease[5] * 100:.2f}% '
          f'| Lung Lesion: {accuracy_disease[6] * 100:.2f}% | Lung Opacity: {accuracy_disease[7] * 100:.2f}%')
    print(f'\tNo Finding: {accuracy_disease[8] * 100:.2f}% | Pleural Effusion: {accuracy_disease[9] * 100:.2f}% '
          f'| Pleural Other: {accuracy_disease[10] * 100:.2f}% | Pneumonia: {accuracy_disease[11] * 100:.2f}%')
    print(f'\tPneumothorax: {accuracy_disease[12] * 100:.2f}% | Support Devices: {accuracy_disease[13] * 100:.2f}%')
    print('\nIndividual sensitivity scores:')
    print(f'\tAtelectasis: {sensitivity_disease[0] * 100:.2f}% | Cardiomegaly: {sensitivity_disease[1] * 100:.2f}% '
          f'| Consolidation: {sensitivity_disease[2] * 100:.2f}% | Edema: {sensitivity_disease[3] * 100:.2f}%')
    print(f'\tEnlarged Cardiomediastinum: {sensitivity_disease[4] * 100:.2f}% | Fracture: {sensitivity_disease[5] * 100:.2f}% '
          f'| Lung Lesion: {sensitivity_disease[6] * 100:.2f}% | Lung Opacity: {sensitivity_disease[7] * 100:.2f}%')
    print(f'\tNo Finding: {sensitivity_disease[8] * 100:.2f}% | Pleural Effusion: {sensitivity_disease[9] * 100:.2f}% '
          f'| Pleural Other: {sensitivity_disease[10] * 100:.2f}% | Pneumonia: {sensitivity_disease[11] * 100:.2f}%')
    print(f'\tPneumothorax: {sensitivity_disease[12] * 100:.2f}% | Support Devices: {sensitivity_disease[13] * 100:.2f}%')
    print('\nIndividual specifity scores:')
    print(f'\tAtelectasis: {specifity_disease[0] * 100:.2f}% | Cardiomegaly: {specifity_disease[1] * 100:.2f}% '
          f'| Consolidation: {specifity_disease[2] * 100:.2f}% | Edema: {specifity_disease[3] * 100:.2f}%')
    print(f'\tEnlarged Cardiomediastinum: {specifity_disease[4] * 100:.2f}% | Fracture: {specifity_disease[5] * 100:.2f}% '
          f'| Lung Lesion: {specifity_disease[6] * 100:.2f}% | Lung Opacity: {specifity_disease[7] * 100:.2f}%')
    print(f'\tNo Finding: {specifity_disease[8] * 100:.2f}% | Pleural Effusion: {specifity_disease[9] * 100:.2f}% '
          f'| Pleural Other: {specifity_disease[10] * 100:.2f}% | Pneumonia: {specifity_disease[11] * 100:.2f}%')
    print(f'\tPneumothorax: {specifity_disease[12] * 100:.2f}% | Support Devices: {specifity_disease[13] * 100:.2f}%')
    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    mesg = f'\n\n----------------------------------------------------------------------------------------\n' \
           f'\tTotal Accuracy: {accuracy_disease.mean() * 100:.2f}% | Total sensitivity: {sensitivity_disease.mean() * 100:.2f}%' \
           f' | Total specifity: {specifity_disease.mean() * 100:.2f}%' \
           f'\n\nIndividual Accuracy scores:' \
           f'\tAtelectasis: {accuracy_disease[0] * 100:.2f}% | Cardiomegaly: {accuracy_disease[1] * 100:.2f}% ' \
          f'| Consolidation: {accuracy_disease[2] * 100:.2f}% | Edema: {accuracy_disease[3] * 100:.2f}%' \
           f'\tEnlarged Cardiomediastinum: {accuracy_disease[4] * 100:.2f}% | Fracture: {accuracy_disease[5] * 100:.2f}% ' \
           f'| Lung Lesion: {accuracy_disease[6] * 100:.2f}% | Lung Opacity: {accuracy_disease[7] * 100:.2f}%' \
        f'\tNo Finding: {accuracy_disease[8] * 100:.2f}% | Pleural Effusion: {accuracy_disease[9] * 100:.2f}% ' \
           f'| Pleural Other: {accuracy_disease[10] * 100:.2f}% | Pneumonia: {accuracy_disease[11] * 100:.2f}%' \
           f'\tPneumothorax: {accuracy_disease[12] * 100:.2f}% | Support Devices: {accuracy_disease[13] * 100:.2f}%' \
           f'\n\nIndividual sensitivity scores:' \
           f'\tAtelectasis: {sensitivity_disease[0] * 100:.2f}% | Cardiomegaly: {sensitivity_disease[1] * 100:.2f}% ' \
          f'| Consolidation: {sensitivity_disease[2] * 100:.2f}% | Edema: {sensitivity_disease[3] * 100:.2f}%' \
           f'\tEnlarged Cardiomediastinum: {sensitivity_disease[4] * 100:.2f}% | Fracture: {sensitivity_disease[5] * 100:.2f}% ' \
           f'| Lung Lesion: {sensitivity_disease[6] * 100:.2f}% | Lung Opacity: {sensitivity_disease[7] * 100:.2f}%' \
        f'\tNo Finding: {sensitivity_disease[8] * 100:.2f}% | Pleural Effusion: {sensitivity_disease[9] * 100:.2f}% ' \
           f'| Pleural Other: {sensitivity_disease[10] * 100:.2f}% | Pneumonia: {sensitivity_disease[11] * 100:.2f}%' \
           f'\tPneumothorax: {sensitivity_disease[12] * 100:.2f}% | Support Devices: {sensitivity_disease[13] * 100:.2f}%' \
           f'\n\nIndividual specifity scores:' \
           f'\tAtelectasis: {specifity_disease[0] * 100:.2f}% | Cardiomegaly: {specifity_disease[1] * 100:.2f}% ' \
          f'| Consolidation: {specifity_disease[2] * 100:.2f}% | Edema: {specifity_disease[3] * 100:.2f}%' \
           f'\tEnlarged Cardiomediastinum: {specifity_disease[4] * 100:.2f}% | Fracture: {specifity_disease[5] * 100:.2f}% ' \
           f'| Lung Lesion: {specifity_disease[6] * 100:.2f}% | Lung Opacity: {specifity_disease[7] * 100:.2f}%' \
        f'\tNo Finding: {specifity_disease[8] * 100:.2f}% | Pleural Effusion: {specifity_disease[9] * 100:.2f}% ' \
           f'| Pleural Other: {specifity_disease[10] * 100:.2f}% | Pneumonia: {specifity_disease[11] * 100:.2f}%' \
           f'\tPneumothorax: {specifity_disease[12] * 100:.2f}% | Support Devices: {specifity_disease[13] * 100:.2f}%'
    with open(os.path.join(params['target_dir'], params['stat_log_path'], '/test_results'), 'a') as f:
        f.write(mesg)






if __name__ == '__main__':
    # delete_experiment(experiment_name='first_try', global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml")
    main_train_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml",
                  valid=True, resume=False, augment=False, experiment_name='xception_p10-11_weight_14_labels_2e5')
    # main_test_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", experiment_name='first_try')
