from __future__ import print_function
import pdb
import argparse
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import random
from functions import (
    training as training_module,
    test as test_module,
    DatasetCorrupterDetect as corrupter,
)
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
from sklearn.datasets import fetch_openml
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from cleanlab.filter import find_label_issues
import struct
import warnings
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)


def main() -> None:
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--all_corruption", required=True, nargs="*", help="array of corutppino rates"
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--dataset", help="dataset to use", required=True)
    parser.add_argument("--model_architecture", default="MLP", help="Model to use")
    parser.add_argument("--num_classes", required=True, help="Number of output classes")
    parser.add_argument("--corruption_rate", required=True, help="corruption rate")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of epochs to run for"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--base_folder", required=True, help="Where to store everything"
    )
    parser.add_argument("--jobid", help="jobid", required=True)
    args = parser.parse_args()
    # Download and load the training data

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    use_mps = not args.no_mps and torch.backends.mps.is_available()

    device: torch.device = torch.device(
        "cuda" if use_cuda else "mps" if use_mps else "cpu"
    )

    dataset_train: None
    dataset_test: None
    new_labels: List[int]
    corruption_tracker: List[bool]
    datasetImgDimensions: int = None
    numColourChannels: int = None
    numClasses: int = None

    if args.dataset == DatasetType.MNIST.value:
        dataset_train = datasets.MNIST(
            root="../data", download=True, train=True, transform=transform
        )
        dataset_test = datasets.MNIST(
            root="../data", download=True, train=False, transform=transform
        )
        datasetImgDimensions = 28
        numColourChannels = 1
        numClasses = 10
    elif args.dataset == DatasetType.CleanMNIST.value:
        dataset_train = datasets.CleanMNIST(
            root="../data", download=True, train=True, transform=transform
        )
        dataset_test = datasets.CleanMNIST(
            root="../data", download=True, train=False, transform=transform
        )
        datasetImgDimensions = 28
        numColourChannels = 1
        numClasses = 10
    elif args.dataset == DatasetType.KMNIST.value:
        dataset_train = datasets.KMNIST(
            root="../data", download=True, train=True, transform=transform
        )
        dataset_test = datasets.KMNIST(
            root="../data", download=True, train=False, transform=transform
        )
        datasetImgDimensions = 28
        numColourChannels = 1
        numClasses = 10

    new_labels, corruption_tracker = corrupter.corrupt_data(
        dataset_train, 1, float(args.corruption_rate), 10
    )
    # Corruption tracker is a boolean array indicating whether a label has been corrupted or not (1 is corrupted, 0 not)

    # These are the old labels
    old_labels: List[int] = dataset_train.targets

    # Setting the targets of the training dataset to the new labels
    dataset_train.targets = new_labels
    # print("here are the new labels", dataset_train.targets)

    # Here is the array tracking what has been corrupted
    # print("corruption tracker", corruption_tracker)

    # Here are the images
    # print("images", dataset_train.data)

    # ---------------------------------------------------------------------------#
    # David's Implementation Here

    # Set a seed for reproducibility
    def setSeed():
        SEED = 321  # any constant value
        np.random.seed(SEED)  # using the same seed for numpy and torch
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(SEED)
        warnings.filterwarnings(
            "ignore", "Lazy modules are a new feature.*"
        )  # ignore warning related to lazy modules

    # Define a classification model
    # currently using baisc model, can swap to resnet50
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

    # Compute out-of-sample predicted probabilities
    def outOfSamplePredProbs(ClassifierModule, datasets, labels):
        model_skorch = NeuralNetClassifier(ClassifierModule)
        num_crossval_folds = (
            3  # for efficiency; values like 5 or 10 will generally work better
        )
        pred_probs = cross_val_predict(
            model_skorch,
            datasets,
            labels,
            cv=num_crossval_folds,
            n_jobs=-1,
            method="predict_proba",
        )
        predicted_labels = pred_probs.argmax(axis=1)
        acc = accuracy_score(labels, predicted_labels)
        return pred_probs, predicted_labels, acc

    # find the label issues
    # def findLabelIssues(labels, pred_probs):
    #     ranked_label_issues = find_label_issues(
    #         labels, pred_probs, return_indices_ranked_by="self_confidence", frac_noise=1
    #     )
    #     return ranked_label_issues

    # reformat data from tensor to ndarray
    def formatDataset(dataset_data, dataset_labels):
        formatted_data = dataset_data.data.numpy().astype(
            "float32"
        )  # 2D array (images are flattened into 1D)

        formatted_data = formatted_data.reshape(
            len(formatted_data), 1, 28, 28
        )  # reshape into [N, C, H, W] for PyTorch

        formatted_labels = dataset_labels.numpy().astype("int64")  # 1D array of labels
        return formatted_data, formatted_labels

    # prune bad data from dataset
    def pruneDataset(data_train, data_labels, label_issues):
        mislabel_indices = label_issues
        # mislabel_indices.sort(reverse=True)
        pruned_data = np.delete(data_train, mislabel_indices, axis=0)
        pruned_labels = np.delete(data_labels, mislabel_indices, axis=0)
        return pruned_data, pruned_labels

    # find correctly identified mislabeled instances
    def compareLabelIssues(corruptor_indices, label_issue_indices):
        common_indices = np.intersect1d(corruptor_indices, label_issue_indices)
        return common_indices

    # helper function to plot and data
    def plot_examples(id_iter, nrows=1, ncols=1):
        for count, id in enumerate(id_iter):
            plt.subplot(nrows, ncols, count + 1)
            plt.imshow(dataset_data[id].reshape(28, 28), cmap="gray")
            plt.title(f"id: {id} \n label: {dataset_labels[id]}")
            plt.axis("off")

        plt.tight_layout(h_pad=2.0)

    setSeed()
    model = setModel()
    dataset_data, dataset_labels = formatDataset(dataset_train.data, new_labels)

    pred_probabilities, predicted_labels, acc = outOfSamplePredProbs(
        model, dataset_data, dataset_labels
    )

    """
    pruning filter options:
    'prune_by_class', 
    'prune_by_noise_rate',
    'Both',
    'confident_learning', 
    'predicted_neq_given', 
    'low_normalized_margin', 
    'low_self_confidence'   
    Default:  'prune_by_noise_rate')
    """
    pruning_filter = "prune_by_noise_rate"

    label_issues = find_label_issues(
        dataset_labels,
        pred_probabilities,
        return_indices_ranked_by="self_confidence",
        filter_by=pruning_filter,  # change filter here
        frac_noise=1,
    )
    print(f"Label issues: {label_issues}")

    amountCorrupted = sum(corruption_tracker)
    corrupted_labels = np.where(corruption_tracker)
    overlap = compareLabelIssues(corrupted_labels, label_issues)

    print("corrupted:", amountCorrupted)
    print("detected:", len(label_issues))
    print(label_issues)
    print("Overlapping indices:", len(overlap))

    pruned_data, pruned_labels = pruneDataset(
        dataset_data, dataset_labels, label_issues
    )
    tensor_data = torch.from_numpy(pruned_data)
    tensor_labels = torch.from_numpy(pruned_labels)

    # convert into DataSet object for dataLoader
    torched_dataset = TensorDataset(tensor_data, tensor_labels)
    print(f"length of pruned data: {len(tensor_data)}")
    print(f"length of pruned label: {len(tensor_data)}")

    # plot examples of label issues
    # plot_examples(label_issues[range(20)], 4, 5)

    # ---------Nolen's comparison integration----------------

    model = None

    if args.model_architecture == ModelType.MLP.value:
        model = SimpleMLP(
            image_dimensions=datasetImgDimensions,
            inputChannels=numColourChannels,
            output_size=numClasses,
        )
    elif args.model_architecture == ModelType.CNN.value:
        model = ConvNN(
            image_dimensions=datasetImgDimensions,
            input_channels=numColourChannels,
            output_size=numClasses,
        )

    model.to_device(device)

    dataPath = "../../data"

    # if args.dataset == DatasetType.MNIST.value:
    #     dataset_train = datasets.MNIST(
    #         dataPath, train=True, download=True, transform=transform
    #     )
    #     dataset_test = datasets.MNIST(dataPath, train=False, transform=transform)
    #     datasetImgDimensions = 28
    #     numColourChannels = 1
    #     numClasses = 10

    if int(args.num_classes) != -1:
        numClasses = args.num_classes

    # Filter the dataset for the specified classes via boolean indexing
    selected_classes: torch.Tensor = torch.arange(numClasses)
    # Creates a boolean array, takin the labels from the dataset, and returning a boolean for if that label is in the selected_classes
    class_filter_train: List[bool] = [
        label in selected_classes for label in dataset_train.targets
    ]
    class_filter_test: List[bool] = [
        label in selected_classes for label in dataset_test.targets
    ]

    # dataset_train.targets = torch.tensor(dataset_train.targets)
    # dataset_test.targets = torch.tensor(dataset_test.targets)
    # Set the data to only those with a corresponding true in the boolean array
    # dataset_train.data = dataset_train.data[class_filter_train]
    # dataset_train.targets = dataset_train.targets[class_filter_train]
    # dataset_test.data = dataset_test.data[class_filter_test]
    # dataset_test.targets = dataset_test.targets[class_filter_test]

    train_dataloader = DataLoader(torched_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(torched_dataset, batch_size=64, shuffle=True)

    optimizer: torch.optim.Optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9
    )
    criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss()
    # Set the learning rate scheduler
    scheduler: torch.optim.lr_scheduler._LRScheduler = StepLR(
        optimizer, step_size=1, gamma=args.gamma
    )

    accuracy_arr: List[float] = []
    test_average_loss_arr: List[float] = []
    train_average_loss_arr: List[float] = []
    avg_sens_arr: List[float] = []
    avg_spec_arr: List[float] = []

    for epoch in range(1, args.epochs + 1):
        train_avg_loss: float = training_module.train(
            args,
            model,
            device,
            train_loader_corrupted=train_dataloader,
            train_loader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        test_avg_loss, correct, total_correct, test_sens, test_spec = test_module.test(
            model, device, test_dataloader, args.jobid, num_classes=numClasses
        )
        # Add accuracy, test loss, training loss to array
        accuracy_arr.append(float(int(correct) / int(total_correct)))
        test_average_loss_arr.append(test_avg_loss)
        train_average_loss_arr.append(train_avg_loss)
        avg_sens_arr.append(test_sens)
        avg_spec_arr.append(test_spec)
        # Step the scheduler
        scheduler.step()

    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    # Save the model
    if args.save_model:
        # Define a unique name for the model checkpoint based on the current timestamp
        model_checkpoint_name: str = (
            f"{args.base_folder}/model_params/{model.__class__.__name__}_{timestamp}.pt"
        )
        torch.save(model.state_dict(), model_checkpoint_name)

    # Write corruption rate, accuracy, average loss to the csv

    with open(f"{args.base_folder}/csv_data/data.csv", "a", newline="") as file:
        data_writer: csv.Writer = csv.writer(file)
        # Iterate through arrays,
        for epoch in range(1, args.epochs + 1):
            data_writer.writerow(
                [
                    float(args.corruption_rate),
                    epoch,
                    round(float(accuracy_arr[epoch - 1]), 4) * 100,
                    round(float(train_average_loss_arr[epoch - 1]), 2),
                    round(float(test_average_loss_arr[epoch - 1]), 2),
                    avg_sens_arr[epoch - 1],
                    avg_spec_arr[epoch - 1],
                ]
            )

    data: dict = {
        "corruption_arr": args.all_corruption,
        "model_name": args.model_architecture,
        "creation_date": timestamp,
        "hyperparams": model.get_hyperparams(),
        "optimizer": type(optimizer).__name__,
        "criterion": type(criterion).__name__,
        "scheduler": type(scheduler).__name__,
        "dataset": args.dataset,
        "num_classes": numClasses,
        "num_epochs": args.epochs,
    }
    # Write the the job details to a json for use in versioning
    with open(f"{args.base_folder}/job_details.json", "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()

# python3 prune_sens_analysis.py --seed 1 --dataset "MNIST"
