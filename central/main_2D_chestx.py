"""
Created on Feb 1, 2022.
main_2D_chestx.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from models.Xception import Xception
from config.serde import open_experiment, create_experiment, delete_experiment
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
    loss_function = CrossEntropyLoss
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

    trainer = Training(cfg_path, num_iterations=params['num_iterations'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function, weight=WEIGHT)
    trainer.execute_training(train_loader=train_loader, valid_loader=valid_loader, batch_size=params['Network']['batch_size'])





if __name__ == '__main__':
    delete_experiment(experiment_name='first_try', global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml")
    main_train_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml",
                  valid=False, resume=False, augment=False, experiment_name='first_try')