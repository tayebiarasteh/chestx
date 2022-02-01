"""
Created on Feb 1, 2022.
main_2D_chestx.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import numpy as np
import nibabel as nib
from torch.nn import CrossEntropyLoss
import torch
import os
from tqdm import tqdm
import glob

from models.Xception import Xception
from config.serde import open_experiment, create_experiment, delete_experiment
from Train_Valid_chestx import Training
from Prediction_chestx import Prediction

import warnings
warnings.filterwarnings('ignore')



def main_train_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name'):
    """Main function for training + validation for directly 3d-wise
        This is the dataloader based on our own implementation.

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

        probabilistic: bool
            if True, we are using the full data with label propagation
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
    weight_path = params['file_path']
    weight_path = weight_path.replace('images', 'labels')
    weight_path = os.path.join(weight_path, "train")
    WEIGHT = torch.Tensor(weight_creator(path=weight_path))

    trainer = Training(cfg_path, num_iterations=params['num_iterations'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function, weight=WEIGHT)
    trainer.execute_training(validation=valid, augmentation=augment, batch_size=params['Network']['batch_size'], )



def main_train_3D_epochbased(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name'):
    """Main function for training + validation for directly 3d-wise
        This is the dataloader based on our own implementation.

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

        probabilistic: bool
            if True, we are using the full data with label propagation
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    # Changeable network parameters
    model = UNet3D()
    loss_function = CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    # class weights corresponding to the dataset
    weight_path = params['file_path']
    weight_path = weight_path.replace('images', 'labels')
    weight_path = os.path.join(weight_path, "train")
    WEIGHT = torch.Tensor(weight_creator(path=weight_path))

    train_dataset = Torchbased_Dataset(cfg_path=cfg_path, mode=Mode.TRAIN)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=False, drop_last=True, shuffle=True, num_workers=40)
    if valid:
        valid_dataset = Torchbased_Dataset(cfg_path=cfg_path, mode=Mode.VALIDATION)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=False, drop_last=True, shuffle=False, num_workers=10)
    else:
        valid_loader = None

    trainer = Training(cfg_path, num_iterations=params['num_iterations'], resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function)
    else:
        trainer.setup_model(model=model, optimiser=optimizer,
                        loss_function=loss_function, weight=WEIGHT)
    trainer.execute_training_epochbased(train_loader=train_loader, valid_loader=valid_loader, augment=augment)



def main_full_test_3D(global_config_path="/home/soroosh/Dropbox/Documents/Repositories/chestx/central/config/config.yaml",
                    experiment_name='name'):
    """Prediction without evaluation for all the images.
    This is the dataloader based on our own implementation.

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    model = UNet3D()

    # Initialize prediction
    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model)

    # Generate test set
    test_set = data_provider_3D(cfg_path, train=False)
    file_list = glob.glob(os.path.join(os.path.join(params['file_path'], "test"), "*"))
    file_list.sort()

    for idx in tqdm(range(len(file_list))):
        x_input, x_input_nifti, img_resized, scaling_ratio = test_set.provide_test_without_label(file_path=file_list[idx])

        max_preds = predictor.new_predict_3D(x_input) # (d,h,w)
        max_preds = max_preds.cpu().detach().numpy()
        max_preds = max_preds.transpose(0,1,3,4,2) # (n, c, h, w, d)

        x_input_nifti.header['pixdim'][1:4] = scaling_ratio
        x_input_nifti.header['dim'][1:4] = np.array(img_resized.shape)
        x_input_nifti.affine[0, 0] = - scaling_ratio[0]
        x_input_nifti.affine[1, 1] = - scaling_ratio[1]
        x_input_nifti.affine[2, 2] = scaling_ratio[2]

        segmentation = nib.Nifti1Image(max_preds[0,0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(segmentation, os.path.join(os.path.join(params['target_dir'], params['output_data_path']), os.path.basename(file_list[idx]).replace('subvolume', '_downsampled_label')))
        input_img = nib.Nifti1Image(img_resized, affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(input_img, os.path.join(os.path.join(params['target_dir'], params['output_data_path']), os.path.basename(file_list[idx]).replace('subvolume', '_downsampled_image')))





if __name__ == '__main__':
    delete_experiment(experiment_name='temppone_phase_THeart', global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml")
    main_train_3D_epochbased(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml",
                  valid=False, resume=False, augment=True, experiment_name='temppone_phase_THeart')
    # main_train_3D(global_config_path="/home/soroosh/Documents/Repositories/chestx/central/config/config.yaml",
    #               valid=True, resume=False, augment=False, experiment_name='temppone_phase_THeart')