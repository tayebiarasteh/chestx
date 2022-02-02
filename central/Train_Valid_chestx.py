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
from enum import Enum
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import torchmetrics
import torchio as tio
from sklearn.metrics import multilabel_confusion_matrix

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15



class Training:
    def __init__(self, cfg_path, num_iterations=10, resume=False, torch_seed=None):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        num_iterations: int
            Total number of iterations for training

        resume: bool
            if we are resuming training from a checkpoint

        torch_seed: int
            Seed used for random generators in PyTorch functions
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.num_iterations = num_iterations

        if resume == False:
            self.model_info = self.params['Network']
            self.model_info['seed'] = torch_seed or self.model_info['seed']
            self.iteration = 0
            self.best_F1 = float('inf')
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
            torch.cuda.manual_seed_all(self.model_info['seed'])
            torch.manual_seed(self.model_info['seed'])
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
            self.loss_function = loss_function(weight=self.loss_weight)
        else:
            self.loss_function = loss_function()
        self.optimiser = optimiser

        # Saves the model, optimiser,loss function name for writing to config file
        # self.model_info['optimiser'] = optimiser
        # self.model_info['model'] = model.__name__
        self.model_info['total_param_num'] = total_param_num
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['num_iterations'] = self.num_iterations
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
        checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']), self.params['checkpoint_name'])
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
        self.iteration = checkpoint['iteration']
        self.best_F1 = checkpoint['best_F1']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.iteration + 1)


    def execute_training(self, train_loader, valid_loader=None, batch_size=1):
        """Executes training by running training and validation at each epoch.
        This is the pipeline based on Pytorch's Dataset and Dataloader

        Parameters
        ----------
        train_loader: Pytorch dataloader object
            training data loader

        valid_loader: Pytorch dataloader object
            validation data loader
       """
        self.params = read_config(self.cfg_path)
        total_start_time = time.time()

        for iteration in range(self.num_iterations - self.iteration):
            self.iteration += 1
            start_time = time.time()

            train_F1, train_acc, train_loss = self.train_epoch(train_loader, batch_size)
            if not valid_loader == None:
                valid_F1, valid_acc, valid_loss = self.valid_epoch(valid_loader, batch_size)

            # Validation iteration & calculate metrics
            if (self.iteration) % (self.params['network_save_freq']) == 0:
                end_time = time.time()
                iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                # saving the model, checkpoint, TensorBoard, etc.
                if not valid_loader == None:
                    self.calculate_tb_stats(train_F1=train_F1, train_acc=train_acc, train_loss=train_loss,
                                            valid_F1=valid_F1, valid_acc=valid_acc, valid_loss=valid_loss)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_F1, train_acc, train_loss,
                                        valid_F1, valid_acc, valid_loss)
                else:
                    self.calculate_tb_stats(train_F1=train_F1, train_acc=train_acc, train_loss=train_loss)
                    self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                                        total_mins, total_secs, train_F1, train_acc, train_loss)


    def train_epoch(self, train_loader, batch_size):
        """Training epoch

        Returns
        -------
        epoch_f1_score: float
        average training F1 score

        epoch_accuracy: float
            average training accuracy

        epoch_loss: float
            average training loss
        """

        # initializing the loss list
        batch_loss = 0
        batch_count = 0
        previous_idx = 0

        # initializing the caches
        logits_with_sigmoid_cache = torch.from_numpy(np.zeros((len(train_loader) * batch_size, 2)))
        logits_no_sigmoid_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))
        labels_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))

        for idx, (image, label) in enumerate(train_loader):
            image = image.to(self.device)
            label = label.to(self.device)

            self.optimiser.zero_grad()

            with torch.set_grad_enabled(True):

                output = self.model(image)
                label = label.float()
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

                # saving the logits and labels of this batch
                for i, batch in enumerate(output_sigmoided):
                    logits_with_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(output):
                    logits_no_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(label):
                    labels_cache[idx * batch_size + i] = batch

                # Loss
                loss = self.loss_function(output, label)
                batch_loss += loss.item()
                batch_count += 1

                #Backward and optimize
                loss.backward()
                self.optimiser.step()

                # Prints loss statistics after number of steps specified.
                if (idx + 1) % self.params['display_stats_freq'] == 0:
                    print('Epoch {:02} | Batch {:03}-{:03} | Train loss: {:.3f}'.
                          format(self.iteration, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    batch_loss = 0
                    batch_count = 0

        # Metrics calculation (macro) over the whole set
        crack_confusion, inactive_confusion = multilabel_confusion_matrix(labels_cache.cpu(), logits_with_sigmoid_cache.cpu())
        # Crack class
        TN = crack_confusion[0, 0]
        FP = crack_confusion[0, 1]
        FN = crack_confusion[1, 0]
        TP = crack_confusion[1, 1]
        accuracy_Crack = (TP + TN) / (TP + TN + FP + FN + epsilon)
        F1_Crack = 2 * TP / (2 * TP + FN + FP + epsilon)
        # Inactive class
        TN_inactive = inactive_confusion[0, 0]
        FP_inactive = inactive_confusion[0, 1]
        FN_inactive = inactive_confusion[1, 0]
        TP_inactive = inactive_confusion[1, 1]
        accuracy_inactive = (TP_inactive + TN_inactive) / (TP_inactive + TN_inactive + FP_inactive + FN_inactive + epsilon)
        F1_inactive = 2 * TP_inactive / (2 * TP_inactive + FN_inactive + FP_inactive + epsilon)
        # Macro averaging
        epoch_accuracy = (accuracy_Crack + accuracy_inactive) / 2
        epoch_f1_score = (F1_Crack + F1_inactive) / 2
        loss = self.loss_function(logits_no_sigmoid_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        return epoch_f1_score, epoch_accuracy, epoch_loss



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

        previous_idx = 0

        with torch.no_grad():
            # initializing the loss list
            batch_loss = 0
            batch_count = 0

            # initializing the caches
            logits_with_sigmoid_cache = torch.from_numpy(np.zeros((len(valid_loader) * batch_size, 2)))
            logits_no_sigmoid_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))
            labels_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))

            for idx, (image, label) in enumerate(valid_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                label = label.float()
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

                # saving the logits and labels of this batch
                for i, batch in enumerate(output_sigmoided):
                    logits_with_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(output):
                    logits_no_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(label):
                    labels_cache[idx * batch_size + i] = batch

                # Loss
                loss = self.loss_function(output, label)
                batch_loss += loss.item()
                batch_count += 1

                # Prints loss statistics after number of steps specified.
                if (idx + 1) % self.params['display_stats_freq'] == 0:
                    print('Epoch {:02} | Batch {:03}-{:03} | Val. loss: {:.3f}'.
                          format(self.iteration, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    batch_loss = 0
                    batch_count = 0

        # Metrics calculation (macro) over the whole set
        crack_confusion, inactive_confusion = multilabel_confusion_matrix(labels_cache.cpu(), logits_with_sigmoid_cache.cpu())
        # Crack class
        TN = crack_confusion[0, 0]
        FP = crack_confusion[0, 1]
        FN = crack_confusion[1, 0]
        TP = crack_confusion[1, 1]
        accuracy_Crack = (TP + TN) / (TP + TN + FP + FN + epsilon)
        F1_Crack = 2 * TP / (2 * TP + FN + FP + epsilon)
        # Inactive class
        TN_inactive = inactive_confusion[0, 0]
        FP_inactive = inactive_confusion[0, 1]
        FN_inactive = inactive_confusion[1, 0]
        TP_inactive = inactive_confusion[1, 1]
        accuracy_inactive = (TP_inactive + TN_inactive) / (TP_inactive + TN_inactive + FP_inactive + FN_inactive + epsilon)
        F1_inactive = 2 * TP_inactive / (2 * TP_inactive + FN_inactive + FP_inactive + epsilon)
        # Macro averaging
        epoch_accuracy = (accuracy_Crack + accuracy_inactive) / 2
        epoch_f1_score = (F1_Crack + F1_inactive) / 2
        loss = self.loss_function(logits_no_sigmoid_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        return epoch_f1_score, epoch_accuracy, epoch_loss



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs,
                       total_hours, total_mins, total_secs, train_F1, train_acc,
                       train_loss, valid_F1=None, valid_acc=None, valid_loss=None):
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

        valid_loss: float
            validation loss of the model

        train_acc: float
            training accuracy of the model

        valid_acc: float
            validation accuracy of the model

        train_F1: float
            training F1 score of the model

        valid_F1: float
            validation F1 score of the model
        """

        # Saves information about training to config file
        self.params['Network']['num_steps'] = self.iteration
        write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving the model based on the best F1
        if valid_F1:
            if valid_F1 < self.best_F1:
                self.best_F1 = valid_F1
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])
        else:
            if train_F1 < self.best_F1:
                self.best_F1 = train_F1
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])

        # Saving every couple of iterations
        if (self.iteration) % self.params['network_save_freq'] == 0:
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                       'iteration{}_'.format(self.iteration) + self.params['trained_model_name'])

        # Save a checkpoint every 2 iterations
        if (self.iteration) % self.params['network_checkpoint_freq'] == 0:
            torch.save({'iteration': self.iteration,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss_state_dict': self.loss_function.state_dict(), 'num_iterations': self.num_iterations,
                        'model_info': self.model_info, 'best_F1': self.best_F1},
                       os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' + self.params['checkpoint_name'])

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'Iteration: {self.iteration}/{self.num_iterations} | '
              f'Iteration Time: {iteration_hours}h {iteration_mins}m {iteration_secs}s | '
              f'Total Time: {total_hours}h {total_mins}m {total_secs}s')
        print(f'\n\tTrain Loss: {train_loss:.4f} | Acc: {train_acc * 100:.2f}% | F1: {train_F1 * 100:.2f}%')

        if valid_loss:
            print(f'\t Val. Loss: {valid_loss:.4f} | Acc: {valid_acc * 100:.2f}% | F1: {valid_F1 * 100:.2f}%')

            # saving the training and validation stats
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Iteration: {self.iteration}/{self.num_iterations} | Iteration Time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total Time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain Loss: {train_loss:.4f} | ' \
                   f'Acc: {train_acc * 100:.2f}% | ' \
                   f'F1: {train_F1 * 100:.2f}%\n\t Val. Loss: {valid_loss:.4f} | Acc: {valid_acc*100:.2f}% | F1: {valid_F1 * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Iteration: {self.iteration}/{self.num_iterations} | Iteration Time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total Time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain Loss: {train_loss:.4f} | ' \
                   f'Acc: {train_acc * 100:.2f}% | F1: {train_F1 * 100:.2f}%\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, train_F1, train_acc, train_loss, valid_F1=None, valid_acc=None, valid_loss=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        Parameters
        ----------
        train_loss: float
            training loss of the model

        valid_loss: float
            validation loss of the model

        train_acc: float
            training accuracy of the model

        valid_acc: float
            validation accuracy of the model

        train_F1: float
            training F1 score of the model

        valid_F1: float
            validation F1 score of the model
        """

        self.writer.add_scalar('Train_F1', train_F1, self.iteration)
        self.writer.add_scalar('Train_Accuracy', train_acc, self.iteration)
        self.writer.add_scalar('Train_Loss', train_loss, self.iteration)
        if valid_F1 is not None:
            self.writer.add_scalar('Valid_F1', valid_F1, self.iteration)
            self.writer.add_scalar('Valid_Accuracy', valid_acc, self.iteration)
            self.writer.add_scalar('Valid_Loss', valid_loss, self.iteration)



class Mode(Enum):
    """
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    """
    TRAIN = 0
    TEST = 1
    VALIDATION = 2