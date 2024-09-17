from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import random
from functions import training as training_module, test as test_module, DatasetCorrupterDetect as corrupter
from modules.base_network import BaseNetwork
from modules.conv_nn import ConvNN
from modules.multilayer_perceptron import SimpleMLP
from modules.DataEfficientImageTransformer import DataEfficientImageTransformer
from modules.resnet18 import create_ResNet18
from modules.vision_transformer import CustomVisionTransformer
import csv
from enums.modelArchEnum import ModelType
from enums.datasetEnum import DatasetType
import sys
from typing import List, Tuple, Any
import json
import numpy as np
import warnings
from sklearn.datasets import fetch_openml
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from cleanlab.filter import find_label_issues
import struct




def main() -> None:
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', required = True, type = int)
    parser.add_argument('--dataset', help = "dataset to use", required = True)
    args = parser.parse_args()
    # Download and load the training data

    dataset_train: None
    dataset_test: None
    new_labels: List[int]
    corruption_tracker:List[bool]
    if(args.dataset == DatasetType.MNIST.value):

        dataset_train = datasets.MNIST(root='../data', download=True, train=True, transform=transform)
        dataset_test = datasets.MNIST(root = '../data', download = True, train = False, transform = transform)

    new_labels, corruption_tracker = corrupter.corrupt_data(dataset_train, 1, 0.1, 10)
    #Corruption tracker is a boolean array indicating whether a label has been corrupted or not (1 is corrupted, 0 not)




    #These are the old labels
    old_labels: List[int] = dataset_train.targets

    #Setting the targets of the training dataset to the new labels
    dataset_train.targets = new_labels
    print("here are the new labels", dataset_train.targets)

    #Here is the array tracking what has been corrupted
    print("corruption tracker", corruption_tracker)

    #Here are the images
    print("images", dataset_train.data)

    #---------------------------------------------------------------------------#
    #David's Implementation Here




    #Set a seed for reproducibility
    def setSeed():
        SEED = 321 #any constant value
        np.random.seed(SEED)  #using the same seed for numpy and torch
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(SEED)



    #Define a classification model
    #currently using baisc model, can swap to resnet50
    def setModel():
        class ClassifierModule(nn.Module):
            def __init__(self):
                super().__init__()

                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 6, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(6),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, 3),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.out = nn.Sequential(
                    nn.Flatten(),
                    nn.LazyLinear(128),
                    nn.ReLU(),
                    nn.Linear(128, 10),
                    nn.Softmax(dim=-1),
                )

            def forward(self, X):
                X = self.cnn(X)
                X = self.out(X)
                return X
        model = ClassifierModule()
        return model  # Return the created ClassifierModule instance

  #Compute out-of-sample predicted probabilities
    #pass the correct arguments
    def outOfSamplePredProbs(ClassifierModule, datasets, labels):
        model_skorch = NeuralNetClassifier(ClassifierModule)
        num_crossval_folds = 3  # for efficiency; values like 5 or 10 will generally work better
        pred_probs = cross_val_predict(
            model_skorch,
            dataset,
            labels,
            cv=num_crossval_folds,
            method="predict_proba",
        )
        predicted_labels = pred_probs.argmax(axis=1)
        acc = accuracy_score(labels, predicted_labels)
        return pred_probs, predicted_labels, acc


    #find the label issues
    def findLabelIssues(labels ,pred_probs):
        ranked_label_issues = find_label_issues(
        labels,
        pred_probs,
        return_indices_ranked_by="self_confidence",
        frac_noise=1
        )
        return ranked_label_issues

    



    setSeed()    
    model = setModel()

    data_train = dataset_train.data
    data_labels = new_labels

    pred_probs, predicted_labels, acc = outOfSamplePredProbs(model, data_train, data_labels)
    label_issues = findLabelIssues(new_labels, pred_probs)


    #here are the labels that are most likely corrupted
    print(f"Label issues: {label_issues}")

    mislabel_indices = label_issues
    mislabel_indices.sort(reverse=True)
    pruned_dataset = np.delete(data_train, mislabel_indices, axis=0)
    pruned_labels = np.delete(data_labels, mislabel_indices, axis=0)


if __name__ == '__main__':
    main()







