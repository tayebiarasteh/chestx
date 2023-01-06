"""
Created on Feb 1, 2022.
main_2D_chestx.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms, models
import timm
import numpy as np
from sklearn import metrics

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from Train_Valid_chestx import Training
from Train_Valid_chestx_federated import Training_federated
from single_head_Train_Valid_chestx import Training_single_head
from Prediction_chestx import Prediction
from data.data_provider import vindr_data_loader_2D, coronahack_data_loader_2D, chexpert_data_loader_2D, mimic_data_loader_2D, UKA_data_loader_2D, cxr14_data_loader_2D, vindr_data_loader_2D_site1, vindr_data_loader_2D_site2, vindr_data_loader_2D_site3
from data.data_provider_manual import vindr_data_loader_2D_manual, coronahack_data_loader_2D_manual, cxr14_data_loader_2D_manual, chexpert_data_loader_2D_manual, mimic_data_loader_2D_manual, UKA_data_loader_2D_manual, vindr_normalFL_site1_data_loader_2D_manual, vindr_normalFL_site2_data_loader_2D_manual, vindr_normalFL_site3_data_loader_2D_manual

import warnings
warnings.filterwarnings('ignore')




def main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr', singlehead=False, pretrained=False):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'coronahack':
        train_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network']['batch_size'],
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    # model = load_resnet50_5FC(num_classes=len(weight), pretrained=False)
    model = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50, pretrained=pretrained)
    # model = load_pretrained_timm_model(num_classes=len(weight), model_name='vit_base_patch16_224', pretrained=pretrained)

    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names, singlehead=singlehead)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader)



def main_backbone_train_2D_federated_manual_batch(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
                  resume=False, augment=False, experiment_name='name', dataset_names_list='vindr', aggregationweight=[1, 1, 1], HE=False, precision_fractional=15, pretrained=False):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    train_loader = []
    valid_loader = []
    model_loader = []
    weight_loader = []
    loss_function_loader = []
    optimizer_loader = []
    label_names_loader = []

    for dataset in dataset_names_list:
        if dataset == 'vindr':
            train_dataset_model = vindr_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'coronahack':
            train_dataset_model = coronahack_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = coronahack_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
        elif dataset == 'mimic':
            train_dataset_model = mimic_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'UKA':
            train_dataset_model = UKA_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'cxr14':
            train_dataset_model = cxr14_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

        elif dataset == 'vindr_site1':
            train_dataset_model = vindr_normalFL_site1_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

        elif dataset == 'vindr_site2':
            train_dataset_model = vindr_normalFL_site2_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

        elif dataset == 'vindr_site3':
            train_dataset_model = vindr_normalFL_site3_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

        weight_model = train_dataset_model.pos_weight()
        label_names_model = train_dataset_model.chosen_labels

        # model_model = load_resnet50_5FC(num_classes=len(weight_model), pretrained=pretrained)
        # model_model = load_pretrained_model_1FC(num_classes=len(weight_model), resnet_num=50, pretrained=pretrained)
        model_model = load_pretrained_timm_model(num_classes=len(weight_model), model_name='vit_base_patch16_224', pretrained=pretrained)

        loss_function_model = BCEWithLogitsLoss
        optimizer_model = torch.optim.Adam(model_model.parameters(), lr=float(params['Network']['lr']),
                                           weight_decay=float(params['Network']['weight_decay']),
                                           amsgrad=params['Network']['amsgrad'])
        train_loader.append(train_dataset_model)
        model_loader.append(model_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)
        optimizer_loader.append(optimizer_model)
        # if dataset == 'cxr14':
        #     continue
        label_names_loader.append(label_names_model)
        valid_loader_model = torch.utils.data.DataLoader(dataset=valid_dataset_model,
                                                         batch_size=params['Network']['batch_size'],
                                                         pin_memory=True, drop_last=False, shuffle=False, num_workers=4)
        valid_loader.append(valid_loader_model)

    trainer = Training_federated(cfg_path, resume=resume, label_names_loader=label_names_loader)

    if resume == True:
        trainer.load_checkpoint(model_loader=model_loader, optimizer_loader=optimizer_loader, loss_function_loader=loss_function_loader, label_names_loader=label_names_loader, weight_loader=weight_loader)
    else:
        trainer.setup_models(model_loader=model_loader, optimizer_loader=optimizer_loader, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_federated_nosyft(train_loader=train_loader, valid_loader=valid_loader, only_one_batch=True, aggregationweight=aggregationweight)


def main_train_2D_conventional_federated_epoch(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
                  resume=False, augment=False, experiment_name='name', dataset_names_list='vindr', aggregationweight=[1, 1, 1], pretrained=False):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"

        resume: bool
            if we are resuming training on a model

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    train_loader = []
    valid_loader = []
    model_loader = []
    weight_loader = []
    loss_function_loader = []
    optimizer_loader = []
    label_names_loader = []

    for dataset in dataset_names_list:
        if dataset == 'vindr':
            train_dataset_model = vindr_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'coronahack':
            train_dataset_model = coronahack_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = coronahack_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
        elif dataset == 'mimic':
            train_dataset_model = mimic_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'UKA':
            train_dataset_model = UKA_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
        elif dataset == 'cxr14':
            train_dataset_model = cxr14_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            valid_dataset_model = cxr14_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

        elif dataset == 'vindr_site1':
            # train_dataset_model = vindr_normalFL_site1_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            train_dataset_model = vindr_data_loader_2D_site1(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

        elif dataset == 'vindr_site2':
            # train_dataset_model = vindr_normalFL_site2_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            train_dataset_model = vindr_data_loader_2D_site2(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

        elif dataset == 'vindr_site3':
            # train_dataset_model = vindr_normalFL_site3_data_loader_2D_manual(cfg_path=cfg_path, mode='train', augment=augment, batch_size=params['Network']['batch_size'])
            train_dataset_model = vindr_data_loader_2D_site3(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

        weight_model = train_dataset_model.pos_weight()
        label_names_model = train_dataset_model.chosen_labels

        # model_model = load_resnet50_5FC(num_classes=len(weight_model), pretrained=pretrained)
        model_model = load_pretrained_model_1FC(num_classes=len(weight_model), resnet_num=50, pretrained=pretrained)
        # model_model = load_pretrained_timm_model(num_classes=len(weight_model), model_name='vit_base_patch16_224', pretrained=pretrained)

        loss_function_model = BCEWithLogitsLoss
        optimizer_model = torch.optim.Adam(model_model.parameters(), lr=float(params['Network']['lr']),
                                           weight_decay=float(params['Network']['weight_decay']),
                                           amsgrad=params['Network']['amsgrad'])
        train_loader_model = torch.utils.data.DataLoader(dataset=train_dataset_model,
                                                         batch_size=params['Network']['batch_size'],
                                                         pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
        train_loader.append(train_loader_model)
        model_loader.append(model_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)
        optimizer_loader.append(optimizer_model)
        # if dataset == 'cxr14':
        #     continue
        label_names_loader.append(label_names_model)
        valid_loader_model = torch.utils.data.DataLoader(dataset=valid_dataset_model,
                                                         batch_size=params['Network']['batch_size'],
                                                         pin_memory=True, drop_last=False, shuffle=False, num_workers=4)
        valid_loader.append(valid_loader_model)

    trainer = Training_federated(cfg_path, resume=resume, label_names_loader=label_names_loader)

    if resume == True:
        trainer.load_checkpoint(model_loader=model_loader, optimizer_loader=optimizer_loader, loss_function_loader=loss_function_loader, label_names_loader=label_names_loader, weight_loader=weight_loader)
    else:
        trainer.setup_models(model_loader=model_loader, optimizer_loader=optimizer_loader, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_conventional_federated(train_loader=train_loader, valid_loader=valid_loader, only_one_batch=False, aggregationweight=aggregationweight)




def main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml", experiment_name='central_exp_for_test',
                 dataset_name='vindr'):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'coronahack':
        test_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters
    model = load_resnet50_5FC(num_classes=len(weight))
    # model = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50)
    # model = load_pretrained_timm_model(num_classes=len(weight), model_name='vit_base_patch16_224')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model)
    average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision = predictor.evaluate_2D(test_loader)

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\t experiment: {experiment_name}\n')
    print(f'\t model tested on the {dataset_name} test set\n')

    print(f'\t avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}%'
    f' | avg specificity: {average_specificity.mean() * 100:.2f}%'
    f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n')

    print('Individual AUROC:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_AUROC[idx] * 100:.2f}%')

    print('\nIndividual accuracy:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_accuracy[idx] * 100:.2f}%')

    print('\nIndividual sensitivity:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_sensitivity[idx] * 100:.2f}%')

    print('\nIndividual specificity:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_specificity[idx] * 100:.2f}%')

    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t experiment: {experiment_name}\n\n' \
          f'\t model tested on the {dataset_name} test set\n\n' \
          f'avg AUROC: {average_AUROC.mean() * 100:.2f}% | avg accuracy: {average_accuracy.mean() * 100:.2f}% ' \
          f' | avg specificity: {average_specificity.mean() * 100:.2f}%' \
          f' | avg recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | avg precision: {average_precision.mean() * 100:.2f}% | avg F1: {average_f1_score.mean() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)

    msg = f'Individual AUROC:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_AUROC[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual accuracy:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_accuracy[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual sensitivity:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_sensitivity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

    msg = f'\n\nIndividual specificity:\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        msg = f'{pathology}: {average_specificity[idx] * 100:.2f}% | '
        with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)




def main_single_head_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml", valid=False,
                  augment=False, experiment_name='name', dataset_name='vindr', model_file_name='epoch170_model0_trained_model.pth', pretrained=False):
    """
        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml"

        valid: bool
            if we want to do validation

        augment: bool
            if we want to have data augmentation during training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    if dataset_name == 'vindr':
        train_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'coronahack':
        train_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'cxr14':
        train_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

    elif dataset_name == 'vindr_site1':
        train_dataset = vindr_data_loader_2D_site1(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

    elif dataset_name == 'vindr_site2':
        train_dataset = vindr_data_loader_2D_site2(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)

    elif dataset_name == 'vindr_site3':
        train_dataset = vindr_data_loader_2D_site3(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=params['Network_single_head']['batch_size'],
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    label_names = train_dataset.chosen_labels

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=params['Network_single_head']['batch_size'],
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None

    # Changeable network parameters
    # model = load_resnet50_5FC(num_classes=len(weight), pretrained=pretrained)
    model = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50, pretrained=pretrained)
    # model = load_pretrained_timm_model(num_classes=len(weight), model_name='vit_base_patch16_224', pretrained=pretrained)

    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network_single_head']['lr']),
                                 weight_decay=float(params['Network_single_head']['weight_decay']), amsgrad=params['Network_single_head']['amsgrad'])

    trainer = Training_single_head(cfg_path, dataset_name, label_names=label_names)

    trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, model_file_name=model_file_name, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader)



def load_resnet50_5FC(num_classes=2, pretrained=False):
    # Load a pre-trained model from config file

    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 1028), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1028, 1028), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1028, 512), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        torch.nn.Linear(128, num_classes)) # for resnet 50

    return model



def load_pretrained_model_1FC(num_classes=2, resnet_num=34, pretrained=False):
    # Load a pre-trained model from config file

    # Load a pre-trained model from Torchvision
    if resnet_num == 34:
        model = models.resnet34(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = True
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, num_classes))  # for resnet 34

    elif resnet_num == 50:
        model = models.resnet50(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = True
        model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, num_classes)) # for resnet 50

    return model



def load_pretrained_timm_model(num_classes=2, model_name='vit_base_patch16_224', pretrained=False):
    # Load a pre-trained model from config file

    model = timm.create_model(model_name, num_classes=num_classes, img_size=512, pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = True

    return model




def main_test_central_2D_with_bootstrapping(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml", experiment_name='central_exp_for_test',
                 dataset_name='vindr', epoch_num=100):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'coronahack':
        test_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters
    # model = load_resnet50_5FC(num_classes=len(weight))
    model = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50)
    # model = load_pretrained_timm_model(num_classes=len(weight), model_name='vit_base_patch16_224')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model, epoch_num=epoch_num)
    pred_array, target_array = predictor.predict_only(test_loader)

    AUC_list = predictor.bootstrapper(pred_array.cpu().numpy(), target_array.int().cpu().numpy(), index_list)



def main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
                                                 experiment_name1='central_exp_for_test', experiment_name2='central_exp_for_test',
                                                 experiment1_epoch_num=100, experiment2_epoch_num=100, dataset_name='vindr'):
    """Main function for multi label prediction

    Parameters
    ----------
    experiment_name: str
        name of the experiment to be loaded.
    """
    params1 = open_experiment(experiment_name1, global_config_path)
    cfg_path1 = params1['cfg_path']

    if dataset_name == 'vindr':
        test_dataset = vindr_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    elif dataset_name == 'coronahack':
        test_dataset = coronahack_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    elif dataset_name == 'chexpert':
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    elif dataset_name == 'UKA':
        test_dataset = UKA_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    elif dataset_name == 'cxr14':
        test_dataset = cxr14_data_loader_2D(cfg_path=cfg_path1, mode='test', augment=False)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters
    # model1 = load_resnet50_5FC(num_classes=len(weight))
    model1 = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50)
    # model1 = load_pretrained_timm_model(num_classes=len(weight), model_name='vit_base_patch16_224')

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params1['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    index_list = []
    for counter in range(1000):
        index_list.append(np.random.choice(len(test_dataset), len(test_dataset)))

    # Initialize prediction 1
    predictor1 = Prediction(cfg_path1, label_names)
    predictor1.setup_model(model=model1, epoch_num=experiment1_epoch_num)
    pred_array1, target_array1 = predictor1.predict_only(test_loader)
    AUC_list1 = predictor1.bootstrapper(pred_array1.cpu().numpy(), target_array1.int().cpu().numpy(), index_list)

    # Changeable network parameters
    # model2 = load_resnet50_5FC(num_classes=len(weight))
    model2 = load_pretrained_model_1FC(num_classes=len(weight), resnet_num=50)
    # model2 = load_pretrained_timm_model(num_classes=len(weight), model_name='vit_base_patch16_224')

    # Initialize prediction 2
    params2 = open_experiment(experiment_name2, global_config_path)
    cfg_path2 = params2['cfg_path']
    predictor2 = Prediction(cfg_path2, label_names)
    predictor2.setup_model(model=model2, epoch_num=experiment2_epoch_num)
    pred_array2, target_array2 = predictor2.predict_only(test_loader)
    AUC_list2 = predictor2.bootstrapper(pred_array2.cpu().numpy(), target_array2.int().cpu().numpy(), index_list)

    print('individual labels p-values:\n')
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            print(f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2')
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                print(f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1')
            else:
                print(f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label')

    print('\nAvg AUC of labels p-values:\n')
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)
    if ratio1 <= 0.05:
        print(f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average')
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)
        if ratio2 <= 0.05:
            print(f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average')
        else:
            print(f'\tp-value: {ratio1}; models NOT significantly different on average for all labels')


    msg = f'\n\nindividual labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    for idx, pathology in enumerate(label_names):
        counter = AUC_list1[:, idx] > AUC_list2[:, idx]
        ratio1 = (len(counter) - counter.sum()) / len(counter)
        if ratio1 <= 0.05:
            msg = f'\t{pathology} p-value: {ratio1}; model 1 significantly higher AUC than model 2'
        else:
            counter = AUC_list2[:, idx] > AUC_list1[:, idx]
            ratio2 = (len(counter) - counter.sum()) / len(counter)
            if ratio2 <= 0.05:
                msg = f'\t{pathology} p-value: {ratio2}; model 2 significantly higher AUC than model 1'
            else:
                msg = f'\t{pathology} p-value: {ratio1}; models NOT significantly different for this label'

        with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)


    msg = f'\n\nAvg AUC of labels p-values:\n'
    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    avgAUC_list1 = AUC_list1.mean(1)
    avgAUC_list2 = AUC_list2.mean(1)
    counter = avgAUC_list1 > avgAUC_list2
    ratio1 = (len(counter) - counter.sum()) / len(counter)
    if ratio1 <= 0.05:
        msg = f'\tp-value: {ratio1}; model 1 significantly higher AUC than model 2 on average'
    else:
        counter = avgAUC_list2 > avgAUC_list1
        ratio2 = (len(counter) - counter.sum()) / len(counter)
        if ratio2 <= 0.05:
            msg = f'\tp-value: {ratio2}; model 2 significantly higher AUC than model 1 on average'
        else:
            msg = f'\tp-value: {ratio1}; models NOT significantly different on average for all labels'

    with open(os.path.join(params1['target_dir'], params1['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)
    with open(os.path.join(params2['target_dir'], params2['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)




if __name__ == '__main__':
    from data.csv_data_preprocess import normalizer_resizer
    handler2 = normalizer_resizer()
    handler2.padchest_normalizer_resizer()

    # delete_experiment(experiment_name='conventional_federated_3sites_vindrfull_1fc_2labelseach_lr5e5_batch12', global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml")
    # main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #               valid=True, resume=False, augment=True, experiment_name='temp', dataset_name='vindr', pretrained=True)

    # main_backbone_train_2D_federated_manual_batch(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #               resume=False, augment=True, experiment_name='ttttbatchaggreg_imagenetpretrain_vindr5k_UKAfull_vitb16_224_1fc_2labelseach_lr5e5_batch5', dataset_names_list=['vindr', 'UKA'], aggregationweight=[1, 1], pretrained=False)

    # main_train_2D_conventional_federated_epoch(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #               resume=False, augment=True, experiment_name='conventional_federated_3sites_vindrfull_1fc_2labelseach_lr5e5_batch12', dataset_names_list=['vindr_site1', 'vindr_site2', 'vindr_site3'], aggregationweight=[1, 1, 1], pretrained=False)
    # main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name='chexpert5k_5fc_resnet50_lr5e5_batch12_5labels', dataset_name='chexpert')
    # main_test_central_2D_with_bootstrapping(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name='chexpertfull_resnet50_1fc_lr5e5_5labels', dataset_name='mimic', epoch_num=28)
    #
    # main_test_central_2D_with_bootstrapping(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name='mimicfull_central5label_cardiomegaly_effusion_pneumonia_consolidation_nofinding', dataset_name='cxr14', epoch_num=28)

    # main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name1='vindr2k_lr9e5_1fc_from23K_ofUKAfull_7labels', experiment_name2='vindr_new2k_1fc_resnet50_lr5e5_7labels',
    #                                              experiment1_epoch_num=3965, experiment2_epoch_num=970, dataset_name='vindr')

    # main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name1='vindr2k_lr9e5_1fc_from23K_ofUKAfull_7labels', experiment_name2='vindr_2k_1fc_resnet50_lr5e5_7labels',
    #                                              experiment1_epoch_num=3965, experiment2_epoch_num=830, dataset_name='vindr')

    # main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name1='chexpertfull_lr9e5_1fc_from11.5K_ofbig_experiment_5centers_5labels', experiment_name2='cxrfull_resnet50_1fc_lr5e5_5labels',
    #                                              experiment1_epoch_num=34, experiment2_epoch_num=36, dataset_name='cxr14')

    # main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name1='vindrfull_lr9e5_1fc_from12.5K_ofbig_experiment_5centers_5labels', experiment_name2='vindrfull_resnet50_1fc_lr5e5_5labels',
    #                                              experiment1_epoch_num=43, experiment2_epoch_num=82, dataset_name='vindr')

    # main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name1='vindrfull_lr9e5_1fc_from12.5K_ofbig_experiment_5centers_5labels', experiment_name2='UKAfull_resnet50_1fc_lr5e5_5labels',
    #                                              experiment1_epoch_num=43, experiment2_epoch_num=22, dataset_name='UKA')
    #
    # main_test_central_2D_pvalue_out_of_bootstrap(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #                      experiment_name1='vindrfull_lr9e5_1fc_from12.5K_ofbig_experiment_5centers_5labels', experiment_name2='mimicfull_resnet50_1fc_lr5e5_5labels',
    #                                              experiment1_epoch_num=43, experiment2_epoch_num=32, dataset_name='mimic')

    # main_single_head_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #               valid=True, augment=True, experiment_name='batchaggreg_vindr_3sites_each5k_resnet50_1fc_lr5e5_2labels', dataset_name='vindr_site1', model_file_name='epoch8000_model0_trained_model.pth')