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

from config.serde import open_experiment, create_experiment, delete_experiment, write_config
from models.Xception_model import Xception
from models.resnet_18 import ResNet18
from Train_Valid_chestx import Training
from Train_Valid_chestx_federated import Training_federated
from Prediction_chestx import Prediction
from data.data_provider import vindr_data_loader_2D, coronahack_data_loader_2D, chexpert_data_loader_2D, mimic_data_loader_2D, UKA_data_loader_2D

import warnings
warnings.filterwarnings('ignore')




def main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml", valid=False,
                  resume=False, augment=False, experiment_name='name', dataset_name='vindr'):
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
        valid_dataset = vindr_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'coronahack':
        train_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = coronahack_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'chexpert':
        train_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'mimic':
        train_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
    elif dataset_name == 'UKA':
        train_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
        valid_dataset = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

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
    # not pretrained resnet
    model = load_pretrained_model(num_classes=len(weight), resnet_num=50)
    # model = Xception(num_classes=len(weight))
    # model = ResNet18(n_out_classes=len(weight))
    loss_function = BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']), amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, num_epochs=params['num_epochs'], resume=resume, label_names=label_names)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight, label_names=label_names)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader)



def main_train_2D_federated(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
                  resume=False, augment=False, experiment_name='name', dataset_names_list='vindr', HE=False, precision_fractional=15):
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
            train_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = vindr_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
        elif dataset == 'coronahack':
            train_dataset_model = coronahack_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = coronahack_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
        elif dataset == 'chexpert':
            train_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = chexpert_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
        elif dataset == 'mimic':
            train_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = mimic_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)
        elif dataset == 'UKA':
            train_dataset_model = UKA_data_loader_2D(cfg_path=cfg_path, mode='train', augment=augment)
            valid_dataset_model = UKA_data_loader_2D(cfg_path=cfg_path, mode='valid', augment=False)

        train_loader_model = torch.utils.data.DataLoader(dataset=train_dataset_model,
                                                         batch_size=params['Network']['batch_size'],
                                                         pin_memory=True, drop_last=True, shuffle=True, num_workers=2)
        weight_model = train_dataset_model.pos_weight()
        label_names_model = train_dataset_model.chosen_labels
        valid_loader_model = torch.utils.data.DataLoader(dataset=valid_dataset_model,
                                                         batch_size=params['Network']['batch_size'],
                                                         pin_memory=True, drop_last=False, shuffle=False, num_workers=2)
        model_model = load_resnet50(num_classes=len(weight_model))
        # model_model = ResNet18(n_out_classes=len(weight_model))
        loss_function_model = BCEWithLogitsLoss
        optimizer_model = torch.optim.Adam(model_model.parameters(), lr=float(params['Network']['lr']),
                                           weight_decay=float(params['Network']['weight_decay']),
                                           amsgrad=params['Network']['amsgrad'])
        train_loader.append(train_loader_model)
        valid_loader.append(valid_loader_model)
        model_loader.append(model_model)
        weight_loader.append(weight_model)
        loss_function_loader.append(loss_function_model)
        optimizer_loader.append(optimizer_model)
        label_names_loader.append(label_names_model)

    trainer = Training_federated(cfg_path, num_epochs=params['num_epochs'], resume=resume, label_names_loader=label_names_loader)
    if resume == True:
        trainer.load_checkpoint(model=model_loader, optimiser=optimizer_loader, loss_function=loss_function_loader, weight=weight_loader, label_names_loader=label_names_loader)
    else:
        trainer.setup_models(model_loader=model_loader, optimizer_loader=optimizer_loader, loss_function_loader=loss_function_loader, weight_loader=weight_loader)
    trainer.training_setup_federated(train_loader=train_loader, valid_loader=valid_loader, HE=HE, precision_fractional=precision_fractional)



def main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml", experiment_name='name',
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
        test_dataset = chexpert_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    elif dataset_name == 'mimic':
        test_dataset = mimic_data_loader_2D(cfg_path=cfg_path, mode='test', augment=False)
    weight = test_dataset.pos_weight()
    label_names = test_dataset.chosen_labels

    # Changeable network parameters
    # not pretrained resnet
    model = load_pretrained_model(num_classes=len(weight), resnet_num=50)
    # model = Xception(num_classes=len(weight))
    # model = ResNet18(n_out_classes=len(weight))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=params['Network']['batch_size'],
                                               pin_memory=True, drop_last=False, shuffle=False, num_workers=16)

    # Initialize prediction
    predictor = Prediction(cfg_path, label_names)
    predictor.setup_model(model=model)
    average_f1_score, average_AUROC, average_accuracy, average_specifity, average_sensitivity, average_precision = predictor.evaluate_2D(test_loader)

    print('------------------------------------------------------'
          '----------------------------------')
    print(f'\t experiment: {experiment_name}\n')
    print(f'\t model tested on the {dataset_name} test set\n')

    print(f'\t Average F1: {average_f1_score.mean() * 100:.2f}% | Average AUROC: {average_AUROC.mean() * 100:.2f}% | Average accuracy: {average_accuracy.mean() * 100:.2f}%'
    f' | Average specifity: {average_specifity.mean() * 100:.2f}%'
    f' | Average recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | Average precision: {average_precision.mean() * 100:.2f}%\n')

    print('Individual F1 scores:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_f1_score[idx] * 100:.2f}%')

    print('\nIndividual AUROC:')
    for idx, pathology in enumerate(predictor.label_names):
        print(f'\t{pathology}: {average_AUROC[idx] * 100:.2f}%')

    print('------------------------------------------------------'
          '----------------------------------')

    # saving the stats
    msg = f'----------------------------------------------------------------------------------------\n' \
          f'\t experiment: {experiment_name}\n\n' \
          f'\t model tested on the {dataset_name} test set\n\n' \
          f'Average F1: {average_f1_score.mean() * 100:.2f}% | Average AUROC: {average_AUROC.mean() * 100:.2f}% | Average accuracy: {average_accuracy.mean() * 100:.2f}% ' \
          f' | Average specifity: {average_specifity.mean() * 100:.2f}%' \
          f' | Average recall (sensitivity): {average_sensitivity.mean() * 100:.2f}% | Average precision: {average_precision.mean() * 100:.2f}%\n\n'

    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_Stats', 'a') as f:
        f.write(msg)



def load_resnet50(num_classes=2):
    # Load a pre-trained model from config file
    # self.model.load_state_dict(torch.load(self.model_info['pretrain_model_path']))

    model = models.resnet50(pretrained=False)
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



def load_pretrained_model(num_classes=2, resnet_num=34):
    # Load a pre-trained model from config file
    # self.model.load_state_dict(torch.load(self.model_info['pretrain_model_path']))

    # Load a pre-trained model from Torchvision
    if resnet_num == 34:
        model = models.resnet34(pretrained=False)
        for param in model.parameters():
            param.requires_grad = True
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, num_classes))  # for resnet 34

    elif resnet_num == 50:
        model = models.resnet50(pretrained=False)
        for param in model.parameters():
            param.requires_grad = True
        model.fc = torch.nn.Sequential(
        #     torch.nn.Linear(2048, 1028), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(1028, 1028), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(1028, 512), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        # torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        # torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        # torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
        # torch.nn.Linear(128, num_classes)) # for resnet 50
        torch.nn.Linear(2048, num_classes)) # for resnet 50
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    return model






if __name__ == '__main__':
    delete_experiment(experiment_name='coronahack', global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml")
    main_train_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
                  valid=True, resume=False, augment=True, experiment_name='coronahack', dataset_name='chexpert')
    # main_train_2D_federated(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml",
    #               resume=False, augment=True, experiment_name='tempp', dataset_names_list=['vindr', 'chexpert'], HE=False, precision_fractional=15)
    # main_test_central_2D(global_config_path="/home/soroosh/Documents/Repositories/chestx/config/config.yaml", experiment_name='tempp', dataset_name='chexpert')
