"""
Created on Feb 1, 2022.
Prediction_chestx.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
import os.path
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pandas as pd

from config.serde import read_config

epsilon = 1e-15



class Prediction:
    def __init__(self, cfg_path, label_names):
        """
        This class represents prediction (testing) process similar to the Training class.
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.label_names = label_names
        self.setup_cuda()


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


    def setup_model(self, model, model_file_name=None, epoch_num=100):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)

        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], model_file_name)))
        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch300_" + model_file_name))
        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "epoch" + str(epoch_num) + "_" + model_file_name))


    def predict_only(self, test_loader):
        """Evaluation with metrics epoch
        """
        self.model.eval()

        # initializing the caches
        preds_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for idx, (image, label) in enumerate(tqdm(test_loader)):

            image = image.to(self.device)
            label = label.to(self.device)
            label = label.float()

            with torch.no_grad():
                output = self.model(image)

                output_sigmoided = F.sigmoid(output)

                # saving the logits and labels of this batch
                preds_with_sigmoid_cache = torch.cat((preds_with_sigmoid_cache, output_sigmoided))
                labels_cache = torch.cat((labels_cache, label))

        return preds_with_sigmoid_cache, labels_cache



    def evaluate_2D(self, test_loader):
        """Testing 2D-wise.

        Parameters
        ----------

        Returns
        -------
        """
        self.model.eval()
        total_f1_score = []
        total_AUROC = []
        total_accuracy = []
        total_specificity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        # initializing the caches
        preds_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for idx, (image, label) in enumerate(tqdm(test_loader)):

            image = image.to(self.device)
            label = label.to(self.device)
            label = label.float()

            with torch.no_grad():
                output = self.model(image)

                output_sigmoided = F.sigmoid(output)

                # saving the logits and labels of this batch
                preds_with_sigmoid_cache = torch.cat((preds_with_sigmoid_cache, output_sigmoided))
                labels_cache = torch.cat((labels_cache, label))

        ############ Evaluation metric calculation ########

        # threshold finding for metrics calculation
        preds_with_sigmoid_cache = preds_with_sigmoid_cache.cpu().numpy()
        labels_cache = labels_cache.int().cpu().numpy()
        optimal_threshold = np.zeros(labels_cache.shape[1])

        for idx in range(labels_cache.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(labels_cache[:, idx], preds_with_sigmoid_cache[:, idx], pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold[idx] = thresholds[optimal_idx]

            metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            plt.annotate('working point', xy=(fpr[optimal_idx], tpr[optimal_idx]), xycoords='data',
                         arrowprops=dict(facecolor='red'))
            plt.grid()
            plt.title(self.label_names[idx] + f' | threshold: {optimal_threshold[idx]:.4f}')
            plt.savefig(os.path.join(self.params['target_dir'], self.params['stat_log_path'], self.label_names[idx] + '.png'))

        predicted_labels = (preds_with_sigmoid_cache > optimal_threshold).astype(np.int32)

        # Metrics calculation (macro) over the whole set
        confusion = metrics.multilabel_confusion_matrix(labels_cache, predicted_labels)

        F1_disease = []
        accuracy_disease = []
        specificity_disease = []
        sensitivity_disease = []
        precision_disease = []

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
        total_AUROC.append(metrics.roc_auc_score(labels_cache, preds_with_sigmoid_cache, average=None))
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

        return average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision


    def plot_confusion_matrix(self, all_matrices, target_names=None,
                              title='Confusion matrix', cmap=None, normalize=False):
        """
        given a sklearn confusion matrix (cm), make a nice plot
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      plt.get_cmap('jet') or plt.cm.Blues
        normalize:    If False, plot the raw numbers
                      If True, plot the proportions
        """

        for cm in all_matrices:
            accuracy = np.trace(cm) / np.sum(cm).astype('float')
            misclass = 1 - accuracy

            if cmap is None:
                cmap = plt.get_cmap('Blues')

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            # plt.imshow(cm)
            plt.title(title)
            plt.colorbar()

            if target_names is not None:
                tick_marks = np.arange(len(target_names))
                plt.xticks(tick_marks, target_names, rotation=45)
                plt.yticks(tick_marks, target_names)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            thresh = cm.max() / 1.5 if normalize else cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label\naccuracy={:0.2f}%; misclass={:0.2f}%'.format(accuracy*100, misclass*100))
            plt.show()


    def bootstrapper(self, preds_with_sigmoid, targets, index_list):
        self.model.eval()
        AUC_list = []
        accuracy_list = []
        specificity_list = []
        sensitivity_list = []
        F1_list = []

        print('bootstrapping ... \n')

        for counter in range(1000):

            final_targets = np.zeros_like(targets)
            final_preds_with_sigmoid = np.zeros_like(preds_with_sigmoid)

            for idx in range(preds_with_sigmoid.shape[-1]):
                new_targets = np.zeros_like(targets[:, idx])
                new_preds_with_sigmoid = np.zeros_like(preds_with_sigmoid[:, idx])
                for i, index in enumerate(index_list[counter]):
                    new_targets[i] = targets[:, idx][index]
                    new_preds_with_sigmoid[i] = preds_with_sigmoid[:, idx][index]

                final_targets[:, idx] = new_targets
                final_preds_with_sigmoid[:, idx] = new_preds_with_sigmoid

            ############ Evaluation metric calculation ########

            # threshold finding for metrics calculation
            optimal_threshold = np.zeros(final_targets.shape[1])

            for idx in range(final_targets.shape[1]):
                fpr, tpr, thresholds = metrics.roc_curve(final_targets[:, idx], final_preds_with_sigmoid[:, idx],
                                                         pos_label=1)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold[idx] = thresholds[optimal_idx]

            predicted_labels = (final_preds_with_sigmoid > optimal_threshold).astype(np.int32)

            # Metrics calculation (macro) over the whole set
            confusion = metrics.multilabel_confusion_matrix(final_targets, predicted_labels)

            F1_disease = []
            accuracy_disease = []
            specificity_disease = []
            sensitivity_disease = []

            for idx, disease in enumerate(confusion):
                TN = disease[0, 0]
                FP = disease[0, 1]
                FN = disease[1, 0]
                TP = disease[1, 1]
                F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
                accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
                specificity_disease.append(TN / (TN + FP + epsilon))
                sensitivity_disease.append(TP / (TP + FN + epsilon))

            average_f1_score = np.stack(F1_disease)
            average_AUROC = np.stack(metrics.roc_auc_score(final_targets, final_preds_with_sigmoid, average=None))
            average_accuracy = np.stack(accuracy_disease)
            average_specificity = np.stack(specificity_disease)
            average_sensitivity = np.stack(sensitivity_disease)

            AUC_list.append(average_AUROC)
            accuracy_list.append(average_accuracy)
            specificity_list.append(average_specificity)
            sensitivity_list.append(average_sensitivity)
            F1_list.append(average_f1_score)

        AUC_list = np.stack(AUC_list)
        accuracy_list = np.stack(accuracy_list)
        specificity_list = np.stack(specificity_list)
        sensitivity_list = np.stack(sensitivity_list)
        F1_list = np.stack(F1_list)

        print('------------------------------------------------------'
              '----------------------------------')
        print('\t experiment:' + self.params['experiment_name'] + '\n')

        print(f'\t avg AUROC: {AUC_list.mean():.2f} ± {AUC_list.std():.2f} | avg accuracy: {accuracy_list.mean():.2f} ± {accuracy_list.std():.2f}'
              f' | avg specificity: {specificity_list.mean():.2f} ± {specificity_list.std():.2f}'
              f' | avg recall (sensitivity): {sensitivity_list.mean():.2f} ± {sensitivity_list.std():.2f} | avg F1: {F1_list.mean():.2f} ± {F1_list.std():.2f}\n')

        print('Individual AUROC:')
        for idx, pathology in enumerate(self.label_names):
            print(f'\t{pathology}: {AUC_list[:, idx].mean():.2f} ± {AUC_list[:, idx].std():.2f}')

        print('\nIndividual accuracy:')
        for idx, pathology in enumerate(self.label_names):
            print(f'\t{pathology}: {accuracy_list[:, idx].mean():.2f} ± {accuracy_list[:, idx].std():.2f}')

        print('\nIndividual sensitivity:')
        for idx, pathology in enumerate(self.label_names):
            print(f'\t{pathology}: {sensitivity_list[:, idx].mean():.2f} ± {sensitivity_list[:, idx].std():.2f}')

        print('\nIndividual specificity:')
        for idx, pathology in enumerate(self.label_names):
            print(f'\t{pathology}: {specificity_list[:, idx].mean():.2f} ± {specificity_list[:, idx].std():.2f}')

        print('------------------------------------------------------'
              '----------------------------------')

        # saving the stats
        msg = f'\n\n----------------------------------------------------------------------------------------\n' \
              '\t experiment:' + self.params['experiment_name'] + '\n\n' \
              f'avg AUROC: {AUC_list.mean():.2f} ± {AUC_list.std():.2f} | avg accuracy: {accuracy_list.mean():.2f} ± {accuracy_list.std():.2f} ' \
              f' | avg specificity: {specificity_list.mean():.2f} ± {specificity_list.std():.2f}' \
              f' | avg recall (sensitivity): {sensitivity_list.mean():.2f} ± {sensitivity_list.std():.2f} | avg F1: {F1_list.mean():.2f} ± {F1_list.std():.2f}\n\n'

        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)

        msg = f'Individual AUROC:\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {AUC_list[:, idx].mean():.2f} ± {AUC_list[:, idx].std():.2f} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual accuracy:\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {accuracy_list[:, idx].mean():.2f} ± {accuracy_list[:, idx].std():.2f} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual sensitivity:\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {sensitivity_list[:, idx].mean():.2f} ± {sensitivity_list[:, idx].std():.2f} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
                f.write(msg)

        msg = f'\n\nIndividual specificity:\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
            f.write(msg)
        for idx, pathology in enumerate(self.label_names):
            msg = f'{pathology}: {specificity_list[:, idx].mean():.2f} ± {specificity_list[:, idx].std():.2f} | '
            with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_Stats', 'a') as f:
                f.write(msg)

        df = pd.DataFrame(AUC_list.mean(1), columns=['AUC_mean'])
        for idx in range(AUC_list.shape[-1]):
            df.insert(idx + 1, 'AUC_' + str(idx + 1), AUC_list[:, idx])

        df.to_csv(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/bootstrapped_AUC_results.csv', sep=',', index=False)

        return AUC_list