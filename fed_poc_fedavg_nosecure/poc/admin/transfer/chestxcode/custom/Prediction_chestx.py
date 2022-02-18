"""
Created on Feb 1, 2022.
Prediction_chestx.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import torch
from sklearn.metrics import multilabel_confusion_matrix
import os.path
import torch.nn.functional as F
import numpy as np

from config.serde import read_config

epsilon = 1e-15



class Prediction:
    def __init__(self, cfg_path):
        """
        This class represents prediction (testing) process similar to the Training class.
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
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


    def setup_model(self, model, model_file_name=None):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)

        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "/" + model_file_name))
        # self.model_p.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "iteration300_" + model_file_name))



    def evaluate_2D(self, test_loader, batch_size):
        """Testing 2D-wise.

        Parameters
        ----------

        Returns
        -------
        F1_disease: float array
            average validation F1 score

        accuracy_disease: float array
            average validation accuracy
        """
        self.model.eval()

        # initializing the metrics lists
        accuracy_disease = []
        F1_disease = []

        with torch.no_grad():

            # initializing the caches
            logits_with_sigmoid_cache = torch.from_numpy(np.zeros((len(test_loader) * batch_size, 14)))
            logits_no_sigmoid_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))
            labels_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))

            for idx, (image, label) in enumerate(test_loader):
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
        confusion = multilabel_confusion_matrix(labels_cache.cpu(), logits_with_sigmoid_cache.cpu())

        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))

        return np.array(accuracy_disease), np.array(F1_disease)
