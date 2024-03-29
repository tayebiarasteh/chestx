"""
Created on May 24, 2022.
Training_Valid_chestx_federated.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import os.path
import time
import pdb
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import syft as sy
from sklearn import metrics
import copy

from config.serde import read_config, write_config

import warnings
warnings.filterwarnings('ignore')
epsilon = 1e-15
hook = sy.TorchHook(torch)



class Training_federated:
    def __init__(self, cfg_path, resume=False, label_names_loader=None):
        """This class represents training and validation processes.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        resume: bool
            if we are resuming training from a checkpoint
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.label_names_loader = label_names_loader

        if resume == False:
            self.model_info = self.params['Network']
            self.epoch = 0
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
            elapsed_secs = elapsed_time - (elapsed_mins * 60)
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_models(self, model_loader, optimizer_loader, loss_function_loader, weight_loader=None):
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

        print('----------------------------------------------------\n')

        self.model_loader = []
        self.optimizer_loader = []
        self.loss_function_loader = []
        for index in range(len(model_loader)):
            self.model_loader.append(model_loader[index].to(self.device))
            self.loss_function_loader.append(loss_function_loader[index](pos_weight=weight_loader[index].to(self.device)))
            self.optimizer_loader.append(optimizer_loader[index])

        # Saves the model, optimiser,loss function name for writing to config file
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)


    def load_checkpoint(self, model_loader, optimizer_loader, loss_function_loader, label_names_loader, weight_loader=None):
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
        self.device = None
        self.setup_cuda()

        self.model_loader = []
        self.optimizer_loader = []
        self.loss_function_loader = []
        for index in range(len(model_loader)):
            checkpoint = torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                                 'model' + str(index) + '_' + self.params['checkpoint_name']))
            self.model_info = checkpoint['model_info']
            self.model_loader.append(model_loader[index].to(self.device))
            self.model_loader[index].load_state_dict(checkpoint['model_state_dict'])
            self.loss_function_loader.append(loss_function_loader[index](pos_weight=weight_loader[index].to(self.device)))
            self.optimizer_loader.append(optimizer_loader[index])

        self.label_names_loader = label_names_loader
        self.epoch = checkpoint['epoch']
        self.writer = SummaryWriter(log_dir=os.path.join(os.path.join(
            self.params['target_dir'], self.params['tb_logs_path'])), purge_step=self.epoch + 1)


    def training_setup_federated(self, train_loader, valid_loader=None, only_one_batch=False, aggregationweight=[1, 1, 1]):
        """
        Parameters
        ----------
        train_loader
        valid_loader
        """
        self.params = read_config(self.cfg_path)

        client_list = []

        for idx in range(len(train_loader)):
            # create a couple workers
            client_list.append(sy.VirtualWorker(hook, id="client" + str(idx)))
        secure_worker = sy.VirtualWorker(hook, id="secure_worker")

        if len(train_loader) == 2:
            client_list[0].add_workers([client_list[1], secure_worker])
            client_list[1].add_workers([client_list[0], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1]])

        elif len(train_loader) == 3:
            client_list[0].add_workers([client_list[1], client_list[2], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2]])

        elif len(train_loader) == 4:
            client_list[0].add_workers([client_list[1], client_list[2], client_list[3], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], client_list[3], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], client_list[3], secure_worker])
            client_list[3].add_workers([client_list[0], client_list[1], client_list[2], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2], client_list[3]])

        elif len(train_loader) == 5:
            client_list[0].add_workers([client_list[1], client_list[2], client_list[3], client_list[4], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], client_list[3], client_list[4], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], client_list[3], client_list[4], secure_worker])
            client_list[3].add_workers([client_list[0], client_list[1], client_list[2], client_list[4], secure_worker])
            client_list[4].add_workers([client_list[0], client_list[1], client_list[2], client_list[3], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[4]])

        elif len(train_loader) == 6:
            client_list[0].add_workers([client_list[1], client_list[2], client_list[3], client_list[4], client_list[5], secure_worker])
            client_list[1].add_workers([client_list[0], client_list[2], client_list[3], client_list[4], client_list[5], secure_worker])
            client_list[2].add_workers([client_list[0], client_list[1], client_list[3], client_list[4], client_list[5], secure_worker])
            client_list[3].add_workers([client_list[0], client_list[1], client_list[2], client_list[4], client_list[5], secure_worker])
            client_list[4].add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[5], secure_worker])
            client_list[5].add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], secure_worker])
            secure_worker.add_workers([client_list[0], client_list[1], client_list[2], client_list[3], client_list[4], client_list[5]])

        ############# copying model state dict names
        self.backbone_state_dict_list = []
        for name in self.model_loader[0].state_dict():
            if 'fc.' or 'head.' in name:
                continue
            self.backbone_state_dict_list.append(name)

        model_state_dict_list_loader = []
        for idx in range(len(self.model_loader)):
            model_state_dict_list = []
            for name in self.model_loader[idx].state_dict():
                model_state_dict_list.append(name)
            model_state_dict_list_loader.append(model_state_dict_list)
        ############# [done] copying model state dict names

        total_start_time = time.time()
        total_overhead_time = 0
        total_datacopy_time = 0

        for epoch in range(self.params['Network']['num_epochs'] - self.epoch):
            self.epoch += 1

            start_time = time.time()
            epoch_overhead_time = 0
            epoch_datacopy_time = 0

            secure_worker.clear_objects()
            new_model_client_list = []
            loss_client_list = []

            for idx in range(len(train_loader)):
                communication_start_time = time.time()
                client_list[idx].clear_objects()
                model = self.model_loader[idx].copy().send(client_list[idx])
                total_overhead_time += (time.time() - communication_start_time)
                epoch_overhead_time += (time.time() - communication_start_time)
                optimizer_model = torch.optim.Adam(model.parameters(), lr=float(self.params['Network']['lr']),
                                                   weight_decay=float(self.params['Network']['weight_decay']),
                                                   amsgrad=self.params['Network']['amsgrad'])

                if only_one_batch:
                    new_model_client, loss_client, overhead = self.train_batch_federated(train_loader[idx], optimizer_model, model, self.loss_function_loader[idx])
                else:
                    new_model_client, loss_client, overhead = self.train_epoch_federated(train_loader[idx], optimizer_model, model, self.loss_function_loader[idx])
                total_datacopy_time += overhead
                epoch_datacopy_time += overhead
                new_model_client_list.append(new_model_client)
                loss_client_list.append(loss_client)

            communication_start_time = time.time()

            ############# copying backbone state dict weights and biases

            temp_dict = {}
            for idx in range(len(train_loader)):
                new_model_client_list[idx].move(secure_worker)

            for weightbias in self.backbone_state_dict_list:
                temp_weight_list = []
                for idx in range(len(train_loader)):
                    temp_weight_list.append(new_model_client_list[idx].state_dict()[weightbias] * aggregationweight[idx])
                temp_dict[weightbias] = (sum(temp_weight_list) / sum(aggregationweight)).clone().get()

            ############# [done] copying backbone state dict weights and biases

            ############# copying model state dict weights and biases
            for idx, model_state_dict_list in enumerate(model_state_dict_list_loader):

                temp_dict_model = {}
                for weightbias in model_state_dict_list:
                    if 'fc.' or 'head.' in weightbias:
                        temp_dict_model[weightbias] = new_model_client_list[idx].state_dict()[weightbias].clone().get()
                    else:
                        temp_dict_model[weightbias] = temp_dict[weightbias]
                self.model_loader[idx].load_state_dict(temp_dict_model)

            ############# [done] copying model state dict weights and biases

            total_overhead_time += (time.time() - communication_start_time)
            epoch_overhead_time += (time.time() - communication_start_time)

            epoch_overhead_hours, epoch_overhead_mins, epoch_overhead_secs = self.time_duration(0, epoch_overhead_time)
            epoch_datacopy_hours, epoch_datacopy_mins, epoch_datacopy_secs = self.time_duration(0, epoch_datacopy_time)
            total_datacopy_hours, total_datacopy_mins, total_datacopy_secs = self.time_duration(0, total_datacopy_time)

            # train loss just as an average of client losses
            train_loss = sum(loss_client_list) / len(loss_client_list)

            # Prints train loss after number of steps specified.
            end_time = time.time()
            iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
            total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

            print('------------------------------------------------------'
                  '----------------------------------')
            print(f'train epoch {self.epoch} | time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s',
                  f'| total: {total_hours}h {total_mins}m {total_secs:.2f}s | epoch communication overhead time: {epoch_overhead_hours}h {epoch_overhead_mins}m {epoch_overhead_secs:.2f}s '
                  f'\nepoch data copying time: {epoch_datacopy_hours}h {epoch_datacopy_mins}m {epoch_datacopy_secs:.2f}s '
                  f'| total data copying time: {total_datacopy_hours}h {total_datacopy_mins}m {total_datacopy_secs:.2f}s\n')

            for idx in range(len(train_loader)):
                print('loss client{}: {:.3f}'.format((idx + 1), loss_client_list[idx]))
                self.writer.add_scalar('Train_loss_client' + str(idx + 1), loss_client_list[idx], self.epoch)

            # Saves information about training to config file
            self.params['Network']['num_epoch'] = self.epoch
            write_config(self.params, self.cfg_path, sort_keys=True)

            ######## Save a checkpoint every epoch ########
            for idx in range(len(self.model_loader)):
                torch.save({'epoch': self.epoch,
                            'model_state_dict': self.model_loader[idx].state_dict(),
                            'optimizer_state_dict': self.optimizer_loader[idx].state_dict(),
                            'loss_state_dict': self.loss_function_loader[idx].state_dict(),
                            'model_info': self.model_info},
                           os.path.join(self.params['target_dir'], self.params['network_output_path'],
                                        'model' + str(idx) + '_' + self.params['checkpoint_name']))
            ######## Save a checkpoint every epoch ########

            # Validation iteration & calculate metrics
            if (self.epoch) % (self.params['display_stats_freq']) == 0:

                # saving the model, checkpoint, TensorBoard, etc.
                valid_loss = []
                valid_accuracy = []
                valid_F1 = []
                valid_AUC = []
                valid_specificity = []
                valid_sensitivity = []
                valid_precision = []

                for idx in range(len(valid_loader)):
                    epoch_loss, average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision, optimal_threshold = self.valid_epoch(
                        valid_loader[idx], self.model_loader[idx], self.loss_function_loader[idx])
                    valid_loss.append(epoch_loss)
                    valid_F1.append(average_f1_score)
                    valid_AUC.append(average_AUROC)
                    valid_accuracy.append(average_accuracy)
                    valid_specificity.append(average_specificity)
                    valid_sensitivity.append(average_sensitivity)
                    valid_precision.append(average_precision)

                    end_time = time.time()
                    total_time = end_time - total_start_time
                    iteration_hours, iteration_mins, iteration_secs = self.time_duration(start_time, end_time)
                    total_hours, total_mins, total_secs = self.time_duration(total_start_time, end_time)

                self.calculate_tb_stats(valid_loss=valid_loss, valid_F1=valid_F1, valid_AUC=valid_AUC, valid_accuracy=valid_accuracy, valid_specificity=valid_specificity,
                                            valid_sensitivity=valid_sensitivity, valid_precision=valid_precision)
                self.savings_prints(iteration_hours, iteration_mins, iteration_secs, total_hours, total_mins,
                                    total_secs, train_loss, total_time, total_overhead_time, total_datacopy_time,
                                    valid_loss=valid_loss, valid_F1=valid_F1, valid_AUC=valid_AUC, valid_accuracy=valid_accuracy,
                                    valid_specificity=valid_specificity, valid_sensitivity=valid_sensitivity, valid_precision=valid_precision, optimal_thresholds=optimal_threshold)



    def train_batch_federated(self, train_loader, optimizer, model, loss_function):
        """Training iteration for only one batch
        """

        model.train()
        image, label = train_loader.provide_mixed()

        communication_start_time = time.time()
        loc = model.location
        image = image.send(loc)
        label = label.send(loc)
        epoch_datacopy = (time.time() - communication_start_time)
        image = image.to(self.device)
        label = label.to(self.device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            output = model(image)
            loss_client = loss_function(output, label)
            loss_client.backward()
            optimizer.step()

        loss_client = loss_client.get().data

        return model, loss_client.item(), epoch_datacopy



    def valid_epoch(self, valid_loader, model, loss_function):
        """Validation epoch

        -------
        """
        model.eval()
        total_f1_score = []
        total_AUROC = []
        total_accuracy = []
        total_specificity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        # initializing the caches
        preds_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        logits_for_loss_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)


        for idx, (image, label) in enumerate(valid_loader):

            image = image.to(self.device)
            label = label.to(self.device)
            # label = label.float()

            with torch.no_grad():
                output = model(image)

                output_sigmoided = F.sigmoid(output)

                # saving the logits and labels of this batch
                preds_with_sigmoid_cache = torch.cat((preds_with_sigmoid_cache, output_sigmoided))
                logits_for_loss_cache = torch.cat((logits_for_loss_cache, output))
                labels_cache = torch.cat((labels_cache, label))

        ############ Evaluation metric calculation ########

        loss = loss_function(logits_for_loss_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        # threshold finding for metrics calculation
        preds_with_sigmoid_cache = preds_with_sigmoid_cache.cpu().numpy()
        labels_cache = labels_cache.int().cpu().numpy()
        optimal_threshold = np.zeros(labels_cache.shape[1])

        for idx in range(labels_cache.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(labels_cache[:, idx], preds_with_sigmoid_cache[:, idx], pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold[idx] = thresholds[optimal_idx]

            # metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            # plt.annotate('working point', xy=(fpr[optimal_idx], tpr[optimal_idx]), xycoords='data',
            #              arrowprops=dict(facecolor='red'))
            # plt.grid()
            # plt.title(self.label_names[idx] + f' | threshold: {optimal_threshold[idx]:.4f} | epoch: {self.epoch}')
            # plt.savefig(self.label_names[idx] + '.png')

        predicted_labels = (preds_with_sigmoid_cache > optimal_threshold).astype(np.int32)

        F1_disease = []
        accuracy_disease = []
        specificity_disease = []
        sensitivity_disease = []
        precision_disease = []

        if labels_cache.shape[-1] == 1:
            confusion = metrics.confusion_matrix(labels_cache, predicted_labels)

            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            TP = confusion[1, 1]
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            specificity_disease.append(TN / (TN + FP + epsilon))
            sensitivity_disease.append(TP / (TP + FN + epsilon))
            precision_disease.append(TP / (TP + FP + epsilon))

        else:
            confusion = metrics.multilabel_confusion_matrix(labels_cache, predicted_labels)

            for idx, disease in enumerate(confusion):
                TN = disease[0, 0]
                FP = disease[0, 1]
                FN = disease[1, 0]
                TP = disease[1, 1]
                F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
                accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
                specificity_disease.append(TN / (TN + FP + epsilon))
                sensitivity_disease.append(TP / (TP + FN + epsilon))
                precision_disease.append(TP / (TP + FP + epsilon))

        # Macro averaging
        total_f1_score.append(np.stack(F1_disease))
        try:
            total_AUROC.append(metrics.roc_auc_score(labels_cache, preds_with_sigmoid_cache, average=None))
        except:
            print('hi')
            pass
        total_accuracy.append(np.stack(accuracy_disease))
        total_specificity_score.append(np.stack(specificity_disease))
        total_sensitivity_score.append(np.stack(sensitivity_disease))
        total_precision_score.append(np.stack(precision_disease))

        average_f1_score = np.stack(total_f1_score).mean(0)
        average_AUROC = np.stack(total_AUROC).mean(0)
        average_accuracy = np.stack(total_accuracy).mean(0)
        average_specificity = np.stack(total_specificity_score).mean(0)
        average_sensitivity = np.stack(total_sensitivity_score).mean(0)
        average_precision = np.stack(total_precision_score).mean(0)

        return epoch_loss, average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision, optimal_threshold



    def savings_prints(self, iteration_hours, iteration_mins, iteration_secs, total_hours,
                       total_mins, total_secs, train_loss, total_time, total_overhead_time=0, total_datacopy_time=0, valid_loss=None, valid_F1=None, valid_AUC=None, valid_accuracy=None,
                       valid_specificity=None, valid_sensitivity=None, valid_precision=None, optimal_thresholds=None):
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
        """

        # Saves information about training to config file
        self.params['Network']['num_epoch'] = self.epoch
        write_config(self.params, self.cfg_path, sort_keys=True)

        overhead_hours, overhead_mins, overhead_secs = self.time_duration(0, total_overhead_time)
        noncopy_time = total_time - total_datacopy_time
        netto_time = total_time - total_overhead_time - total_datacopy_time
        noncopy_hours, noncopy_mins, noncopy_secs = self.time_duration(0, noncopy_time)
        netto_hours, netto_mins, netto_secs = self.time_duration(0, netto_time)

        for idx in range(len(self.model_loader)):

            # Saving every couple of epochs
            if (self.epoch) % self.params['display_stats_freq'] == 0:
                torch.save(self.model_loader[idx].state_dict(), os.path.join(self.params['target_dir'], self.params['network_output_path'],
                           'epoch{}_'.format(self.epoch) + 'model' + str(idx) + '_' + self.params['trained_model_name']))

            # Save a checkpoint every epoch
            torch.save({'epoch': self.epoch,
                        'model_state_dict': self.model_loader[idx].state_dict(),
                        'optimizer_state_dict': self.optimizer_loader[idx].state_dict(),
                        'loss_state_dict': self.loss_function_loader[idx].state_dict(),
                        'model_info': self.model_info},
                       os.path.join(self.params['target_dir'], self.params['network_output_path'], 'model' + str(idx) + '_' + self.params['checkpoint_name']))

            try:
                print('------------------------------------------------------'
                      '----------------------------------')
                print('\t model number:', str(idx))
                print(f'epoch: {self.epoch} | '
                      f'epoch time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s | '
                      f'total time: {total_hours}h {total_mins}m {total_secs:.2f}s | communication overhead time so far: {overhead_hours}h {overhead_mins}m {overhead_secs:.2f}s')
                print(f'\n\tTrain loss: {train_loss:.4f}')

                print(f'\t Val. loss: {valid_loss[idx]:.4f} | avg AUROC: {valid_AUC[idx].mean() * 100:.2f}% | avg accuracy: {valid_accuracy[idx].mean() * 100:.2f}%'
                f' | avg specificity: {valid_specificity[idx].mean() * 100:.2f}%'
                f' | avg recall (sensitivity): {valid_sensitivity[idx].mean() * 100:.2f}% | avg F1: {valid_F1[idx].mean() * 100:.2f}%n')

                print('Individual AUROC:')
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    try:
                        print(f'\t{pathology}: {valid_AUC[idx][i] * 100:.2f}%')
                    except:
                        print(f'\t{pathology}: {valid_AUC[idx] * 100:.2f}%')

                print('\nIndividual accuracy:')
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    print(f'\t{pathology}: {valid_accuracy[idx][i] * 100:.2f}% ; threshold: {optimal_thresholds[idx]:.4f}')

                print('\nIndividual sensitivity:')
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    print(f'\t{pathology}: {valid_sensitivity[idx][i] * 100:.2f}%')

                print('\nIndividual specificity:')
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    print(f'\t{pathology}: {valid_specificity[idx][i] * 100:.2f}%')

                # saving the training and validation stats
                msg = f'\n\n----------------------------------------------------------------------------------------\n' \
                       f'epoch: {self.epoch} | epoch Time: {iteration_hours}h {iteration_mins}m {iteration_secs:.2f}s' \
                       f' | total time: {total_hours}h {total_mins}m {total_secs:.2f}s | ' \
                      f'communication overhead time so far: {overhead_hours}h {overhead_mins}m {overhead_secs:.2f}s\n' \
                      f' | total time - copy time: {noncopy_hours}h {noncopy_mins}m {noncopy_secs:.2f}s' \
                      f' | total time - copy time - overhead time: {netto_hours}h {netto_mins}m {netto_secs:.2f}s' \
                      f'\n\n\tTrain loss: {train_loss:.4f} | ' \
                       f'Val. loss: {valid_loss[idx]:.4f} | avg AUROC: {valid_AUC[idx].mean() * 100:.2f}% | avg accuracy: {valid_accuracy[idx].mean() * 100:.2f}% ' \
                       f' | avg specificity: {valid_specificity[idx].mean() * 100:.2f}%' \
                       f' | avg recall (sensitivity): {valid_sensitivity[idx].mean() * 100:.2f}% | avg precision: {valid_precision[idx].mean() * 100:.2f}% | avg F1: {valid_F1[idx].mean() * 100:.2f}%\n\n'
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                    f.write(msg)

                msg = f'\Individual AUROC:\n'
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                    f.write(msg)
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    try:
                        msg = f'{pathology}: {valid_AUC[idx][i] * 100:.2f}% | '
                    except:
                        msg = f'{pathology}: {valid_AUC[idx] * 100:.2f}% | '
                    with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                        f.write(msg)

                msg = f'\n\nIndividual accuracy:\n'
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                    f.write(msg)
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    msg = f'{pathology}: {valid_accuracy[idx][i] * 100:.2f}% ; threshold: {optimal_thresholds[idx]:.4f} | '
                    with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                        f.write(msg)

                msg = f'\n\nIndividual specificity:\n'
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                    f.write(msg)
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    msg = f'{pathology}: {valid_specificity[idx][i] * 100:.2f}% | '
                    with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                        f.write(msg)

                msg = f'\n\nIndividual sensitivity:\n'
                with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                    f.write(msg)
                for i, pathology in enumerate(self.label_names_loader[idx]):
                    msg = f'{pathology}: {valid_sensitivity[idx][i] * 100:.2f}% | '
                    with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/Stats_' + str(idx), 'a') as f:
                        f.write(msg)
            except:
                continue



    def calculate_tb_stats(self, valid_loss=None, valid_F1=None, valid_AUC=None, valid_accuracy=None, valid_specificity=None, valid_sensitivity=None, valid_precision=None):
        """Adds the evaluation metrics and loss values to the tensorboard.

        """
        for idx in range(len(valid_loss)):

            self.writer.add_scalar('Valid_loss_model_' + str(idx), valid_loss[idx], self.epoch)
            self.writer.add_scalar('valid_avg_F1_model_' + str(idx), valid_F1[idx].mean(), self.epoch)
            self.writer.add_scalar('Valid_avg_AUROC_model_' + str(idx), valid_AUC[idx].mean(), self.epoch)

            # for i, pathology in enumerate(self.label_names_loader[idx]):
            #     self.writer.add_scalar('valid_F1_' + pathology, valid_F1[idx][i], self.epoch)

            self.writer.add_scalar('Valid_avg_accuracy_model_' + str(idx), valid_accuracy[idx].mean(), self.epoch)
            # self.writer.add_scalar('Valid_avg_specificity_model_' + str(idx), valid_specificity[idx].mean(), self.epoch)
            # self.writer.add_scalar('Valid_avg_precision_model_' + str(idx), valid_precision[idx].mean(), self.epoch)
            # self.writer.add_scalar('Valid_avg_recall_sensitivity_model_' + str(idx), valid_sensitivity[idx].mean(), self.epoch)