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
from functions import training as training_module, test as test_module
from modules.base_network import BaseNetwork
from modules.conv_nn import ConvNN
from modules.multilayer_perceptron import SimpleMLP
from modules import DatasetCorrupter as corrupter
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


def main() -> None:
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--all_corruption", nargs="*", help="corruption_rates array")
    parser.add_argument("--manual", type=str, help="Coming from manual run?")
    parser.add_argument(
        "--base_folder", type=str, help="base path to store model parameters"
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        help="type of model architecture being used to train",
        default="CNN",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="dataset on which to test the model",
        default="MNIST",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=-1,
        required=False,
        help="# of classes will be used from the dataset",
    )
    parser.add_argument("--jobid", type=int, help="jobid")
    parser.add_argument(
        "--corruption_rate",
        type=float,
        default=0.1,
        metavar="N",
        help="corruption_rate for training (default 0.1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=13,
        metavar="N",
        help="number of epochs to train (default: 13)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
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
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
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
        "--data_chunk",
        type=int,
        default=1,
        metavar="N",
        help="Chunk to minimize data by (half, quarter, etc.)",
    )
    args = parser.parse_args()

    print("Running with Corruption Rate " + str(args.corruption_rate))
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    # Set seed value
    random.seed(args.seed)
    # Set the random seed for reproducible experiments
    torch.manual_seed(args.seed)
    # Set the device to use
    device: torch.device = torch.device(
        "cuda" if use_cuda else "mps" if use_mps else "cpu"
    )
    print(f"Using device: {device.type}")

    # Set the kwargs for the dataloaders
    train_kwargs: dict = {"batch_size": args.batch_size}
    test_kwargs: dict = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    # Define the transformations to be applied to the data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # based on mean and stdev of MNIST images
        ]
    )
    # Initialize the datasets as None (or with default datasets if appropriate)
    dataset_train: Any = None
    dataset_test: Any = None

    dataPath: str = ""

    isManual: bool = args.manual.lower() == "true"

    if isManual:
        dataPath = "../data"
    else:
        dataPath = "../../data"

    datasetImgDimensions: int = 0
    numColourChannels: int = 0
    numClasses: int = 0
    # Get dataset based on passed in bash script arg
    if args.dataset == DatasetType.MNIST.value:
        dataset_train = datasets.MNIST(
            dataPath, train=True, download=True, transform=transform
        )
        dataset_test = datasets.MNIST(dataPath, train=False, transform=transform)
        datasetImgDimensions = 28
        numColourChannels = 1
        numClasses = 10
    elif args.dataset == DatasetType.FASHIONMNIST.value:
        dataset_train = datasets.FashionMNIST(
            dataPath, train=True, download=True, transform=transform
        )
        dataset_test = datasets.FashionMNIST(dataPath, train=False, transform=transform)
        datasetImgDimensions = 28
        numColourChannels = 1
        numClasses = 10
    elif args.dataset == DatasetType.KMNIST.value:
        dataset_train = datasets.KMNIST(
            dataPath, train=True, download=True, transform=transform
        )
        dataset_test = datasets.KMNIST(dataPath, train=False, transform=transform)
        datasetImgDimensions = 28
        numColourChannels = 1
        numClasses = 10
    elif args.dataset == DatasetType.CIFAR10.value:
        # CIFAR-10 standard transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )
        dataset_train = datasets.CIFAR10(
            dataPath, train=True, download=True, transform=transform
        )
        dataset_test = datasets.CIFAR10(dataPath, train=False, transform=transform)
        datasetImgDimensions = 32
        numColourChannels = 3
        numClasses = 10
    elif args.dataset == DatasetType.CIFAR100.value:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
        dataset_train = datasets.CIFAR100(
            dataPath, train=True, download=True, transform=transform
        )
        dataset_test = datasets.CIFAR100(dataPath, train=False, transform=transform)

        numClasses = 100
        # ImageNet doesn't work right now, need to download separately, then use this
    elif args.dataset == DatasetType.IMAGENET.value:
        # Transformations for training
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Transformations for testing/evaluation
        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        dataset_train = datasets.ImageNet(
            dataPath, train=True, download=True, transform=train_transform
        )
        dataset_test = datasets.ImageNet(
            dataPath, train=False, transform=test_transform
        )

    # Model initialization
    model = None
    # Initialize the model, optimizer, and criterion
    if args.model_architecture == ModelType.CNN.value:
        model: nn.Module = ConvNN(
            image_dimensions=datasetImgDimensions,
            input_channels=numColourChannels,
            output_size=numClasses,
        )
    elif args.model_architecture == ModelType.MLP.value:
        model: nn.Module = SimpleMLP(
            image_dimensions=datasetImgDimensions,
            inputChannels=numColourChannels,
            output_size=numClasses,
        )
    elif args.model_architecture == ModelType.RESNET18.value:
        model: nn.Module = create_ResNet18(
            input_channels=numColourChannels, output_size=numClasses
        )
    elif args.model_architecture == ModelType.VIT.value:
        model: nn.Module = CustomVisionTransformer(
            image_size=datasetImgDimensions,
            num_channels=numColourChannels,
            num_classes=numClasses,
        )
    elif args.model_architecture == ModelType.DEIT.value:
        model: nn.Module = DataEfficientImageTransformer(
            output_size=numClasses,
            image_dimensions=datasetImgDimensions,
            colour_channels=numColourChannels,
        )

    model.to_device(device)

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

    dataset_train.targets = torch.tensor(dataset_train.targets)
    dataset_test.targets = torch.tensor(dataset_test.targets)
    # Set the data to only those with a corresponding true in the boolean array
    dataset_train.data = dataset_train.data[class_filter_train]
    dataset_train.targets = dataset_train.targets[class_filter_train]
    dataset_test.data = dataset_test.data[class_filter_test]
    dataset_test.targets = dataset_test.targets[class_filter_test]
    # Get the total number of samples available in the dataset
    num_samples: int = len(dataset_train)
    # Get list of all indices
    indices: List[int] = list(range(num_samples))
    # Randomly shuffle the list
    random.shuffle(indices)
    # Get the smaller set of indices, flooring to avoid floats
    min_indices: List[int] = indices[: num_samples // args.data_chunk]
    # Get the minimized dataset
    dataset_min: torch.utils.data.Subset = Subset(dataset_train, min_indices)
    # Corrupt data
    dataset_train_corrupted: torch.utils.data.Dataset = corrupter.DatasetCorrupter(
        dataset_min, args.corruption_rate, numClasses, args.seed
    )
    # Setup dataloaders
    train_loader_corrupted: torch.utils.data.DataLoader = DataLoader(
        dataset_train_corrupted, **train_kwargs
    )
    train_loader: torch.utils.data.DataLoader = DataLoader(dataset_min, **train_kwargs)
    test_loader: torch.utils.data.DataLoader = DataLoader(dataset_test, **test_kwargs)
    optimizer: torch.optim.Optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9
    )
    criterion: nn.modules.loss._Loss = nn.CrossEntropyLoss()
    # Set the learning rate scheduler
    scheduler: torch.optim.lr_scheduler._LRScheduler = StepLR(
        optimizer, step_size=1, gamma=args.gamma
    )

    # Initialize variables for loss, accuracy
    # Arrays indexed by epoch

    accuracy_arr: List[float] = []
    test_average_loss_arr: List[float] = []
    train_average_loss_arr: List[float] = []
    avg_sens_arr: List[float] = []
    avg_spec_arr: List[float] = []
    # Iterate over the epochs and call the train and test functions
    for epoch in range(1, args.epochs + 1):
        train_avg_loss: float = training_module.train(
            args,
            model,
            device,
            train_loader_corrupted,
            train_loader,
            optimizer,
            criterion,
            epoch,
        )
        test_avg_loss, correct, total_correct, test_sens, test_spec = test_module.test(
            model, device, test_loader, args.jobid, numClasses
        )
        # Add accuract, test loss, training loss to array
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
        "model_name": args.model_architecture,
        "creation_date": timestamp,
        "hyperparams": model.get_hyperparams(),
        "optimizer": type(optimizer).__name__,
        "criterion": type(criterion).__name__,
        "scheduler": type(scheduler).__name__,
        "dataset": args.dataset,
        "num_classes": numClasses,
        "corruption_arr": args.all_corruption,
        "num_epochs": args.epochs,
        "data_chunk": args.data_chunk,
    }
    # Write the the job details to a json for use in versioning
    with open(f"{args.base_folder}/job_details.json", "w") as file:
        json.dump(data, file, indent=4)


if __name__ == "__main__":
    main()
