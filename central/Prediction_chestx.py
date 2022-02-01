"""
Created on Feb 1, 2022.
Prediction_chestx.py

@author: Soroosh Tayebi Arasteh <sarasteh@ukaachen.de>
https://github.com/tayebiarasteh/
"""

from sklearn import metrics
import pdb
import torch
from math import ceil

from config.serde import *




class Prediction:
    def __init__(self, cfg_path):
        """
        This class represents prediction (testing) process similar to the Training class.
        For both the 3d and 2d- wise segmentation of 3d volumes.
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
        self.model_p = model.to(self.device)

        self.model_p.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "/" + model_file_name))
        # self.model_p.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path']) + "iteration300_" + model_file_name))



    def evaluate_3D(self, batch_size, image, label):
        """Testing 3D-wise.

        Parameters
        ----------
        batch_size: int
            must be dividable by 30 (1, 2, 3, 4, 5, 6, 10, 15, 30)

        image: torch tensor
            images
            (n=30, c=1, d, h, w)

        label: torch tensor
            labels
            (n=30, c=1, d, h, w)

        Returns
        ----------
        f1_score: numpy array of floats (8,)
            individual F1 scores of every label, excluding the background

        accuracy: float
            total accuracy
        """

        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        self.model_p.eval()

        # initializing the caches
        labels_cache = torch.zeros_like(label)
        max_preds_cache = torch.zeros_like(image)

        label = label.long()
        image = image.to(self.device)
        label = label.to(self.device)

        with torch.no_grad():
            for i in range(image.shape[0] // batch_size):
                output = self.model_p(image[i * batch_size:(i + 1) * batch_size])
                max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability (multi-class)

                # caching
                max_preds_cache[i * batch_size:(i + 1) * batch_size] = max_preds
                labels_cache[i * batch_size:(i + 1) * batch_size] = label[i * batch_size:(i + 1) * batch_size]

        accuracy = metrics.accuracy_score(labels_cache.cpu().flatten(), max_preds_cache.cpu().flatten())
        f1_score = metrics.f1_score(labels_cache.cpu().flatten(), max_preds_cache.cpu().flatten(),
                                    labels=[1, 2, 3, 4, 5, 6, 7, 8], average=None)
        precision_score = metrics.precision_score(labels_cache.cpu().flatten(), max_preds_cache.cpu().flatten(),
                                            labels=[1,2,3,4,5,6,7,8], average=None)
        recall_score = metrics.recall_score(labels_cache.cpu().flatten(), max_preds_cache.cpu().flatten(), labels=[1,2,3,4,5,6,7,8], average=None)

        return f1_score, precision_score, recall_score, accuracy



    def predict_3D(self, batch_size, image):
        """Predicting the segmentations 3D-wise.
        No evaluation metric will be calculated here.

        Parameters
        ----------
        batch_size: int
            must be dividable by 30 (1, 2, 3, 4, 5, 6, 10, 15, 30)

        image: torch tensor
            images
            (n=1, c=1, d, h, w)

        Returns
        ----------
        output_tensor: torch tensor
            (n=30, d, h, w)
        """
        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        self.model_p.eval()

        # # initializing the output
        # output_tensor = torch.zeros_like(image)
        # output_tensor = torch.squeeze(output_tensor, dim=1)

        image = image.to(self.device)
        image = image.float()
        output = self.model_p(image)
        max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability (multi-class)
        max_preds = max_preds.cpu().detach().numpy()

        # with torch.no_grad():
        #     for i in range(image.shape[0] // batch_size):
        #         output = self.model_p(image[i * batch_size:(i + 1) * batch_size])
        #         max_preds = output.argmax(dim=1, keepdim=False)  # get the index of the max probability (multi-class)
        #         output_tensor[i * batch_size:(i + 1) * batch_size] = max_preds

        return max_preds


    def new_predict_3D(self, image):
        """Predicting the segmentations 3D-wise.
        No evaluation metric will be calculated here.

        Parameters
        ----------
        image: torch tensor
            images
            (n=1, c=1, d, h, w)

        Returns
        ----------
        output_tensor: torch tensor
            (n=30, d, h, w)
        """
        # Reads params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        self.model_p.eval()

        image = image.to(self.device)
        image = image.float()
        output = self.model_p(image)
        max_preds = output.argmax(dim=1, keepdim=True)  # get the index of the max probability (multi-class)

        return max_preds

