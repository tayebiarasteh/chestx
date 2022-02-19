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
from Train_Valid_chestx import Training
from Prediction_chestx import Prediction
from data.data_provider import data_loader, Mode

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
    model = Xception()
    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    # class weights corresponding to the dataset
    # weight_path = params['file_path']
    # weight_path = weight_path.replace('images', 'labels')
    # weight_path = os.path.join(weight_path, "train")
    # WEIGHT = torch.Tensor(weight_creator(path=weight_path))
    WEIGHT = None

    train_dataset = data_loader(cfg_path=cfg_path, mode=Mode.TRAIN)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=False, drop_last=True, shuffle=True, num_workers=4)
    if valid:
        valid_dataset = data_loader(cfg_path=cfg_path, mode=Mode.VALIDATION)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=False, drop_last=True, shuffle=False, num_workers=1)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function, weight=WEIGHT)
    trainer.execute_training(train_loader=train_loader, valid_loader=valid_loader, batch_size=params['Network']['batch_size'])




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
    model = Xception()

    test_dataset = data_loader(cfg_path=cfg_path, mode=Mode.TEST)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=False, drop_last=True, shuffle=False, num_workers=4)

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)
    accuracy_disease, F1_disease = predictor.evaluate_2D(test_loader, params['Network']['batch_size'])

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\tTotal Accuracy: {accuracy_disease.mean() * 100:.2f}% | Total F1 (dice score): {F1_disease.mean() * 100:.2f}%')
    print('\nIndividual Accuracy scores:')
    print(f'\tAtelectasis: {accuracy_disease[0] * 100:.2f}% | Cardiomegaly: {accuracy_disease[1] * 100:.2f}% '
          f'| Consolidation: {accuracy_disease[2] * 100:.2f}% | Edema: {accuracy_disease[3] * 100:.2f}%')
    print(f'\tEnlarged Cardiomediastinum: {accuracy_disease[4] * 100:.2f}% | Fracture: {accuracy_disease[5] * 100:.2f}% '
          f'| Lung Lesion: {accuracy_disease[6] * 100:.2f}% | Lung Opacity: {accuracy_disease[7] * 100:.2f}%')
    print(f'\tNo Finding: {accuracy_disease[8] * 100:.2f}% | Pleural Effusion: {accuracy_disease[9] * 100:.2f}% '
          f'| Pleural Other: {accuracy_disease[10] * 100:.2f}% | Pneumonia: {accuracy_disease[11] * 100:.2f}%')
    print(f'\tPneumothorax: {accuracy_disease[12] * 100:.2f}% | Support Devices: {accuracy_disease[13] * 100:.2f}%')
    print('\nIndividual F1 scores (dice scores):')
    print(f'\tAtelectasis: {F1_disease[0] * 100:.2f}% | Cardiomegaly: {F1_disease[1] * 100:.2f}% '
          f'| Consolidation: {F1_disease[2] * 100:.2f}% | Edema: {F1_disease[3] * 100:.2f}%')
    print(f'\tEnlarged Cardiomediastinum: {F1_disease[4] * 100:.2f}% | Fracture: {F1_disease[5] * 100:.2f}% '
          f'| Lung Lesion: {F1_disease[6] * 100:.2f}% | Lung Opacity: {F1_disease[7] * 100:.2f}%')
    print(f'\tNo Finding: {F1_disease[8] * 100:.2f}% | Pleural Effusion: {F1_disease[9] * 100:.2f}% '
          f'| Pleural Other: {F1_disease[10] * 100:.2f}% | Pneumonia: {F1_disease[11] * 100:.2f}%')
    print(f'\tPneumothorax: {F1_disease[12] * 100:.2f}% | Support Devices: {F1_disease[13] * 100:.2f}%')
    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    mesg = f'\n\n----------------------------------------------------------------------------------------\n' \
           f'\tTotal Accuracy: {accuracy_disease.mean() * 100:.2f}% | Total F1 (dice score): {F1_disease.mean() * 100:.2f}%' \
           f'\n\nIndividual Accuracy scores:' \
           f'\tAtelectasis: {accuracy_disease[0] * 100:.2f}% | Cardiomegaly: {accuracy_disease[1] * 100:.2f}% ' \
          f'| Consolidation: {accuracy_disease[2] * 100:.2f}% | Edema: {accuracy_disease[3] * 100:.2f}%' \
           f'\tEnlarged Cardiomediastinum: {accuracy_disease[4] * 100:.2f}% | Fracture: {accuracy_disease[5] * 100:.2f}% ' \
           f'| Lung Lesion: {accuracy_disease[6] * 100:.2f}% | Lung Opacity: {accuracy_disease[7] * 100:.2f}%' \
        f'\tNo Finding: {accuracy_disease[8] * 100:.2f}% | Pleural Effusion: {accuracy_disease[9] * 100:.2f}% ' \
           f'| Pleural Other: {accuracy_disease[10] * 100:.2f}% | Pneumonia: {accuracy_disease[11] * 100:.2f}%' \
           f'\tPneumothorax: {accuracy_disease[12] * 100:.2f}% | Support Devices: {accuracy_disease[13] * 100:.2f}%' \
           f'\n\nIndividual F1 scores (dice scores):' \
           f'\tAtelectasis: {F1_disease[0] * 100:.2f}% | Cardiomegaly: {F1_disease[1] * 100:.2f}% ' \
          f'| Consolidation: {F1_disease[2] * 100:.2f}% | Edema: {F1_disease[3] * 100:.2f}%' \
           f'\tEnlarged Cardiomediastinum: {F1_disease[4] * 100:.2f}% | Fracture: {F1_disease[5] * 100:.2f}% ' \
           f'| Lung Lesion: {F1_disease[6] * 100:.2f}% | Lung Opacity: {F1_disease[7] * 100:.2f}%' \
        f'\tNo Finding: {F1_disease[8] * 100:.2f}% | Pleural Effusion: {F1_disease[9] * 100:.2f}% ' \
           f'| Pleural Other: {F1_disease[10] * 100:.2f}% | Pneumonia: {F1_disease[11] * 100:.2f}%' \
           f'\tPneumothorax: {F1_disease[12] * 100:.2f}% | Support Devices: {F1_disease[13] * 100:.2f}%'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results', 'a') as f:
        f.write(mesg)






if __name__ == '__main__':
    delete_experiment(experiment_name='first_try', global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml")
    main_train_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml",
                  valid=True, resume=False, augment=False, experiment_name='first_try')
    # main_test_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", experiment_name='first_try')
