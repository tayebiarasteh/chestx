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
from tqdm import tqdm
import torchmetrics
import torchio as tio
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss

from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.fl_constant import FLContextKey, ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants

from configs.serde import open_experiment, read_config
from models.Xception_model import Xception
from Prediction_chestx import Prediction
from data.data_provider import data_loader, Mode

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15



class Training(Learner):
    def __init__(self, cfg_path, n_local_iterations=5, exclude_vars=None, analytic_sender_id="analytic_sender",
                 valid=False, experiment_name='name'):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        n_local_iterations: int
            Total number of iterations for training

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.

        valid: bool
            if we want to do validation
        """
        super().__init__()
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.exclude_vars = exclude_vars
        self.batch_size = self.params['Network']['batch_size']
        self.valid = valid
        self.experiment_name = experiment_name

        # self.model_info = self.params['Network']
        # self.n_local_iterations = n_local_iterations
        self.step = 0
        self.best_loss = float('inf')
        self.analytic_sender_id = analytic_sender_id


    def initialize(self, parts: dict, fl_ctx: FLContext):
        print('\n\n\n\n\nhey soroosh initialize\n\n\n\n\n\n')

        # params = create_experiment(self.experiment_name, self.cfg_path)
        # cfg_path = params["cfg_path"]

        # Changeable network parameters
        self.model = Xception()
        loss_function = BCEWithLogitsLoss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.params['Network']['lr']),
                                     weight_decay=float(self.params['Network']['weight_decay']),
                                     amsgrad=self.params['Network']['amsgrad'])

        # class weights corresponding to the dataset
        # weight_path = params['file_path']
        # weight_path = weight_path.replace('images', 'labels')
        # weight_path = os.path.join(weight_path, "train")
        # WEIGHT = torch.Tensor(weight_creator(path=weight_path))
        WEIGHT = None

        train_dataset = data_loader(cfg_path=self.cfg_path, mode=Mode.TRAIN)

        # we need a small subset in federated learning
        train_size = int(0.05 * len(train_dataset))
        test_size = len(train_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                   pin_memory=False, drop_last=True, shuffle=True, num_workers=4)
        print('train loader peyda shodd ddddddddddddd', self.train_loader)

        self.n_local_iterations = len(self.train_loader)

        if self.valid:
            valid_dataset = data_loader(cfg_path=self.cfg_path, mode=Mode.VALIDATION)

            # we need a small subset in federated learning
            valid_size = int(0.5 * len(valid_dataset))
            test_size = len(valid_dataset) - valid_size
            valid_dataset, _ = torch.utils.data.random_split(train_dataset, [valid_size, test_size])

            self.valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, pin_memory=False,
                                                            drop_last=True, shuffle=False, num_workers=1)
        self.setup_cuda()
        self.setup_model(optimiser=optimizer, loss_function=loss_function, weight=WEIGHT)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager in case no initial model is found.
        self.default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self.default_train_conf)

        # Tensorboard streaming setup
        self.writer = parts.get(self.analytic_sender_id)  # user configuration from config_fed_client.json
        if not self.writer:  # else use local TensorBoard writer only
            self.writer = SummaryWriter(fl_ctx.get_prop(FLContextKey.APP_ROOT))




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


    def setup_model(self, optimiser, loss_function, weight=None):
        """Setting up all the models, optimizers, and loss functions.

        Parameters
        ----------
        optimiser: optimizer file
            The optimizer

        loss_function: loss file
            The loss function

        weight: 1D tensor of float
            class weights
        """

        # prints the network's total number of trainable parameters and
        # stores it to the experiment config
        total_param_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'\nTotal # of trainable parameters: {total_param_num:,}')
        print('----------------------------------------------------\n')

        if not weight==None:
            self.loss_weight = weight.to(self.device)
            self.loss_function = loss_function(weight=self.loss_weight)
        else:
            self.loss_function = loss_function()
        self.optimiser = optimiser

        # Saves the model, optimiser,loss function name for writing to config file
        # self.model_info['model'] = model.__name__
        # self.model_info['optimiser'] = optimiser.__name__
        # self.model_info['total_param_num'] = total_param_num
        # self.model_info['loss_function'] = loss_function.__name__
        # self.model_info['num_local_iterations'] = self.n_local_iterations
        # self.params['Network'] = self.model_info
        # write_config(self.params, self.cfg_path, sort_keys=True)



    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Executes training by running training and validation at each epoch.
        This is the pipeline based on Pytorch's Dataset and Dataloader

        Parameters
        ----------
        train_loader: Pytorch dataloader object
            training data loader

        valid_loader: Pytorch dataloader object
            validation data loader
       """
        # Get model weights
        try:
            dxo = from_shareable(data)
        except:
            self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Ensure data kind is weights.
        if not dxo.data_kind == DataKind.WEIGHTS:
            self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        # Convert weights to tensor. Run training
        torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
        print('hey soroosh train*************************************************************************************************')
        self.train_epoch(fl_ctx, torch_weights, abort_signal)

        # Check the abort_signal after training.
        # local_train returns early if abort_signal is triggered.
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Save the local model after training.
        self.save_local_model(fl_ctx)

        # Get the new state dict and send as weights
        new_weights = self.model.state_dict()
        new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

        outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self.n_local_iterations})
        return outgoing_dxo.to_shareable()



    def train_epoch(self, fl_ctx, weights, abort_signal):
        """Training epoch
        """

        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        # initializing the loss list
        batch_loss = 0
        batch_count = 0

        total_start_time = time.time()
        start_time = time.time()

        for idx, (image, label) in enumerate(self.train_loader):

            self.model.train()
            if abort_signal.triggered:
                return

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

                # Backward and optimize
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

                    self.log_info(fl_ctx, 'Step {} | batch {} / {} | loss: {:.3f}'.
                          format(self.step, idx + 1, len(self.train_loader), train_loss))
                    self.log_info(fl_ctx, f'\ntime: {iteration_hours}h {iteration_mins}m {iteration_secs}s'
                                          f'| total: {total_hours}h {total_mins}m {total_secs}s\n')
                    self.writer.add_scalar('Train_Loss', train_loss, self.step)

            # Validation iteration & calculate metrics
            if (self.step) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                if self.valid:
                    valid_F1, valid_acc, valid_loss = self.valid_epoch(self.valid_loader, abort_signal)
                    end_time = time.time()
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                    self.calculate_tb_stats(valid_F1=valid_F1, valid_acc=valid_acc, valid_loss=valid_loss)
                #     self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                #                         total_mins, total_secs, train_loss,
                #                         valid_F1, valid_acc, valid_loss)
                # else:
                #     self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours,
                #                         total_mins, total_secs, train_loss)



    def valid_epoch(self, weights, abort_signal):
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

        # initializing the metrics lists
        accuracy_disease = []
        F1_disease = []

        with torch.no_grad():

            # initializing the caches
            logits_with_sigmoid_cache = torch.from_numpy(np.zeros((len(self.valid_loader) * self.batch_size, 14)))
            logits_no_sigmoid_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))
            labels_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))

            for idx, (image, label) in enumerate(self.valid_loader):
                self.model.eval()
                if abort_signal.triggered:
                    return 0

                image = image.to(self.device)
                label = label.to(self.device)
                image = image.float()
                label = label.float()

                output = self.model(image)
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

                # saving the logits and labels of this batch
                for i, batch in enumerate(output_sigmoided):
                    logits_with_sigmoid_cache[idx * self.batch_size + i] = batch
                for i, batch in enumerate(output):
                    logits_no_sigmoid_cache[idx * self.batch_size + i] = batch
                for i, batch in enumerate(label):
                    labels_cache[idx * self.batch_size + i] = batch

        # Metrics calculation (macro) over the whole set
        confusioner = torchmetrics.ConfusionMatrix(num_classes=14, multilabel=True).to(self.device)
        confusion = confusioner(logits_with_sigmoid_cache.to(self.device), labels_cache.int().to(self.device))

        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))

        # Macro averaging
        epoch_accuracy = torch.stack(accuracy_disease).mean().item()
        epoch_f1_score = torch.stack(F1_disease).mean().item()

        loss = self.loss_function(logits_no_sigmoid_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        return epoch_f1_score, epoch_accuracy, epoch_loss



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs,
                       total_hours, total_mins, total_secs,
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
        # self.params['Network']['step'] = self.step
        # write_config(self.params, self.cfg_path, sort_keys=True)

        # Saving the model based on the best loss
        if valid_loss:
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])
        else:
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                           self.params['trained_model_name'])

        # Saving every couple of steps
        if (self.step) % self.params['network_save_freq'] == 0:
            torch.save(self.model.state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' +
                       'step{}_'.format(self.step) + self.params['trained_model_name'])

        # Save a checkpoint every step
        if (self.step) % self.params['network_checkpoint_freq'] == 0:
            torch.save({'step': self.step,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'loss_state_dict': self.loss_function.state_dict(), 'num_local_iterations': self.n_local_iterations,
                        'model_info': self.model_info, 'best_loss': self.best_loss},
                       os.path.join(self.params['target_dir'], self.params['network_output_path']) + '/' + self.params['checkpoint_name'])

        print('------------------------------------------------------'
              '----------------------------------')
        print(f'Step: {self.step} | '
              f'Step time: {iteration_hours}h {iteration_mins}m {iteration_secs}s | '
              f'Total time: {total_hours}h {total_mins}m {total_secs}s')
        print(f'\n\tTrain loss: {train_loss:.4f}')

        if valid_loss:
            print(f'\t Val. loss: {valid_loss:.4f} | Acc: {valid_acc * 100:.2f}% | F1: {valid_F1 * 100:.2f}%')

            # saving the training and validation stats
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Step: {self.step} | Step time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain loss: {train_loss:.4f} | ' \
                   f'Val. loss: {valid_loss:.4f} | Acc: {valid_acc*100:.2f}% | F1: {valid_F1 * 100:.2f}%\n\n'
        else:
            msg = f'----------------------------------------------------------------------------------------\n' \
                   f'Step: {self.step} | Step time: {iteration_hours}h {iteration_mins}m {iteration_secs}s' \
                   f' | Total time: {total_hours}h {total_mins}m {total_secs}s\n\n\tTrain loss: {train_loss:.4f}\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats', 'a') as f:
            f.write(msg)



    def calculate_tb_stats(self, valid_F1=None, valid_acc=None, valid_loss=None):
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
        if valid_F1 is not None:
            self.writer.add_scalar('Valid_F1', valid_F1, self.step)
            self.writer.add_scalar('Valid_Accuracy', valid_acc, self.step)
            self.writer.add_scalar('Valid_Loss', valid_loss, self.step)



    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)


    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self.default_train_conf)
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self.exclude_vars)

        # Get the model parameters and create dxo from it
        dxo = model_learnable_to_dxo(ml)
        return dxo.to_shareable()




class Mode(Enum):
    """
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    """
    TRAIN = 0
    TEST = 1
    VALIDATION = 2