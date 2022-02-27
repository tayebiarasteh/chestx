"""
Created on Feb 1, 2022.
Training_Valid_chestx.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15



class Training:
    def __init__(self, cfg_path, num_epochs=10, resume=False):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        num_epochs: int
            Total number of epochs for training

        resume: bool
            if we are resuming training from a checkpoint
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.num_epochs = num_epochs

        if resume == False:
            self.model_info = self.params['Network']
            self.epoch = 0
            self.step = 0
            self.best_loss = float('inf')
            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['target_dir'], self.params['tb_logs_path']))


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.

        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, optimiser, loss_function, weight=None):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function

        weight: 1D tensor of float
            class weights
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        self.model = model.to(self.device)
        if not weight==None:
            self.loss_weight = weight.to(self.device)
            self.loss_function = loss_function(pos_weight=self.loss_weight)
        else:
            self.loss_function = loss_function()
        self.optimiser = optimiser

        # Saves the model, optimiser,loss function name for writing to config file
        # self.model_info['model'] = model.__name__
        # self.model_info['optimiser'] = optimiser.__name__
        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['num_epochs'] = self.num_epochs
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)


    def load_checkpoint(self, model, optimiser, loss_function):
        """In case of resuming training from a checkpoint,
        loads the weights for all the models, optimizers, and
        loss functions, and device, tensorboard events, number
        of iterations (epochs), and every info from checkpoint.

        Parameters
        ----------
        model: model file
            The network

        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function
        """
        checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                self.params['checkpoint_name']))
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.loss_weight = checkpoint['loss_state_dict']['weight']
        self.loss_weight = self.loss_weight.to(self.device)
        self.loss_function = loss_function(weight=self.loss_weight)
        self.optimiser = optimiser

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.step + 1)



    def train_epoch(self, train_loader, batch_size, valid_loader=None):
        """Training epoch
        """
        self.params = read_config(self.cfg_path)

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1

            # initializing the loss list
            batch_loss = 0
            batch_count = 0

            start_time = time.time()
            total_start_time = time.time()

            for idx, (image, label) in enumerate(train_loader):
                self.model.train()

                image = image.to(self.device)
                label = label.to(self.device)

                self.optimiser.zero_grad()

                with torch.set_grad_enabled(True):

                    image = image.float()
                    label = label.float()

                    output = self.model(image)

                    # Loss
                    loss = self.loss_function(output, label)
                    batch_loss += loss.item()
                    batch_count += 1

                    #Backward and optimize
                    loss.backward()
                    self.optimiser.step()
                    self.step += 1

                    # Prints train loss after number of steps specified.
                    if (self.step) % self.params['display_train_loss_freq'] == 0:
                        end_time = time.time()
                        iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                        total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)
                        train_loss = batch_loss / batch_count
                        batch_loss = 0
                        batch_count = 0
                        start_time = time.time()

                        print('Step {} | train epoch {} | batch {} / {} | loss: {:.3f}'.
                              format(self.step, self.epoch, idx+1, len(train_loader), train_loss),
                              f'\ntime: {iteration_hours}h {iteration_mins}m {iteration_secs}s',
                              f'| total: {total_hours}h {total_mins}m {total_secs}s\n')
                        self.writer.add_scalar('Train_Loss', train_loss, self.step)

                # Validation iteration & calculate metrics
                if (self.step) % (self.params['display_stats_freq']) == 0:

                    # saving the model, checkpoint, TensorBoard, etc.
                    if not valid_loader == None:
                        valid_acc, valid_sensitivity, valid_specifity, valid_loss, valid_F1 = self.valid_epoch(valid_loader, batch_size)
                        end_time = time.time()
                        total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                        self.calculate_tb_stats(valid_acc, valid_sensitivity, valid_specifity, valid_loss, valid_F1)
                        self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                            total_mins, total_secs, train_loss,
                                            valid_acc, valid_sensitivity, valid_specifity, valid_loss, valid_F1)
                    else:
                        self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                            total_mins, total_secs, train_loss)



    def valid_epoch(self, valid_loader, batch_size):
        """Validation epoch

        Returns
        -------
        epoch_f1_score: float
            average validation F1 score

        epoch_accuracy: float
            average validation accuracy

        epoch_loss: float
            average validation loss
        """
        self.model.eval()

        # initializing the metrics lists
        accuracy_disease = []
        sensitivity_disease = []
        specifity_disease = []
        F1_disease = []

        with torch.no_grad():

            # initializing the caches
            logits_with_sigmoid_cache = torch.from_numpy(np.zeros((len(valid_loader) * batch_size, 14)))
            logits_no_sigmoid_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))
            labels_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))

            for idx, (image, label) in enumerate(valid_loader):
                self.model.eval()

                image = image.to(self.device)
                label = label.to(self.device)
                image = image.float()
                label = label.float()

                output = self.model(image)
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

                # saving the logits and labels of this batch
                for i, batch in enumerate(output_sigmoided):
                    logits_with_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(output):
                    logits_no_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(label):
                    labels_cache[idx * batch_size + i] = batch

        # Metrics calculation (macro) over the whole set
        confusioner = torchmetrics.ConfusionMatrix(num_classes=14, multilabel=True).to(self.device)
        confusion = confusioner(logits_with_sigmoid_cache.to(self.device), labels_cache.int().to(self.device))
        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            sensitivity_disease.append(TP / (TP + FN + epsilon))
            specifity_disease.append(TN / (TN + FP + epsilon))
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))

        # Macro averaging
        epoch_accuracy = torch.stack(accuracy_disease).mean().item()
        epoch_sensitivity = torch.stack(sensitivity_disease).mean().item()
        epoch_specifity = torch.stack(specifity_disease).mean().item()
        epoch_f1_score = torch.stack(F1_disease).mean().item()

        loss = self.loss_function(logits_no_sigmoid_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        return epoch_accuracy, epoch_sensitivity, epoch_specifity, epoch_loss, epoch_f1_score



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs,
                       total_hours, total_mins, total_secs, train_loss,
                       valid_acc=None, valid_sensitivity=None, valid_specifity=None, valid_loss=None, valid_F1=None):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        Parameters
        ----------
        iteration_hours: int
            hours part of the elapsed time of each iteration

        iteration_mins: int
            minutes part of the elapsed time of each iteration

        iteration_secs: int
            seconds part of the elapsed time of each iteration

        total_hours: int
            hours part of the total elapsed time

        total_mins: int
            minutes part of the total elapsed time

        total_secs: int
            seconds part of the total elapsed time

        train_loss: float
            training loss of the model

        valid_acc: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specifity: float
            validation specifity of the model

        valid_loss: float
            validation loss of the model
        """

        # Saves information about training to config file
        self.params['Network']['num_epoch'] = self.epoch
        self.params['Network']['step'] = self.step
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving the model based on the best loss
        if valid_loss:
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                                 self.params['network_output_path'], self.params['trained_model_name']))
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'],
                                                                 self.params['network_output_path'], self.params['trained_model_name']))

        # Saving every couple of steps
        if (self.step) % self.params['network_save_freq'] == 0:
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                                             'step{}_'.format(self.step) + self.params['trained_model_name']))

        # Save a checkpoint every step
        if (self.step) % self.params['network_checkpoint_freq'] == 0:
            if self.loss_weight:
                torch.save({'epoch': self.epoch, 'step': self.step, 'weight': self.loss_weight,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimiser.state_dict(),
                            'loss_state_dict': self.loss_function.state_dict(), 'num_epochs': self.num_epochs,
                            'model_info': self.model_info, 'best_loss': self.best_loss},
                           os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))

            else:
                torch.save({'epoch': self.epoch, 'step': self.step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimiser.state_dict(),
                            'loss_state_dict': self.loss_function.state_dict(), 'num_epochs': self.num_epochs,
                            'model_info': self.model_info, 'best_loss': self.best_loss},
                           os.path.join(self.params['target_dir'], self.params['network_output_path'], self.params['checkpoint_name']))

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'Step: {self.step} (epoch: {self.epoch}) | '
              f'Step time: {iteration_hours}h {iteration_mins}m {iteration_secs}s | '
              f'Total time: {total_hours}h {total_mins}m {total_secs}s')
        print(f'\n\tTrain loss: {train_loss:.4f}')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | Acc: {valid_acc * 100:.2f}% | F1: {valid_F1 * 100:.2f}%'
                  f' | Sensitivity: {valid_sensitivity * 100:.2f}% | Specifity: {valid_specifity * 100:.2f}%')

            # saving the training and validation stats
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Step: {self.step} (epoch: {self.epoch}) | Step time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | Acc: {valid_acc*100:.2f}% | F1: {valid_F1 * 100:.2f}% ' \
                  f'| Sensitivity: {valid_sensitivity * 100:.2f}% | Specifity: {valid_specifity * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Step: {self.step} (epoch: {self.epoch}) | Step time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain loss: {train_loss:.4f}\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, valid_acc=None, valid_sensitivity=None, valid_specifity=None, valid_loss=None, valid_F1=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        valid_acc: float
            validation accuracy of the model

        valid_sensitivity: float
            validation sensitivity of the model

        valid_specifity: float
            validation specifity of the model

        valid_loss: float
            validation loss of the model
        """
        if valid_acc is not None:
            self.writer.add_scalar('Valid_Accuracy', valid_acc, self.step)
            self.writer.add_scalar('Valid_Loss', valid_loss, self.step)
            self.writer.add_scalar('Valid_sensitivity', valid_sensitivity, self.step)
            self.writer.add_scalar('Valid_specifity', valid_specifity, self.step)
            self.writer.add_scalar('Valid_F1 score', valid_F1, self.step)
