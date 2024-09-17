import sklearn
import csv
import time
from sklearn.model_selection import train_test_split
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn import datasets
from typing import List
import pandas as pd
import numpy as np
import random
import argparse
from enums.datasetEnum import DatasetType
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--k_neighbours", type=int, help="number of neighbours", default=1
    )
    parser.add_argument("--all_corruption", nargs="*", help="corruption_rates array")
    parser.add_argument("--manual", type=str, help="Coming from manual run?")
    parser.add_argument(
        "--base_folder", type=str, help="base path to store model parameters"
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

    print("Running the KNN algorithm file")
    numClasses: int = args.num_classes
    random.seed(args.seed)
    dataset: None
    if args.dataset == DatasetType.MNIST.value:
        dataset: sklearn.utils.Bunch = datasets.fetch_openml(
            "mnist_784", version=1, cache=True
        )
        numClasses = 10
    elif args.dataset == DatasetType.FASHIONMNIST.value:
        dataset: sklearn.utils.Bunch = datasets.fetch_openml(
            "Fashion-MNIST", version=1, cache=True
        )
        numClasses = 10
    elif args.dataset == DatasetType.KMNIST.value:
        dataset: sklearn.utils.Bunch = datasets.fetch_openml(
            "Kuzushiji-MNIST", version=1, cache=True
        )
        numClasses = 10
    elif args.dataset == DatasetType.CIFAR10.value:
        dataset: sklearn.utils.Bunch = datasets.fetch_openml("CIFAR_10", cache=True)
        numClasses = 10

    if args.num_classes != -1:
        numClasses = args.num_classes

    data, targets = dataset.data, dataset.target.astype(int)

    # Ensure `data` is a numpy array if not already
    data = np.array(data)
    data: np.ndarray = dataset.data
    targets: np.ndarray = dataset.target

    data_train, data_test, targets_train, targets_test = train_test_split(
        data, targets, test_size=0.15, random_state=0, shuffle=True
    )
    # Corrupt the targets_train based on the corruption rate
    # Convert the categorical series to integers directly after splitting
    targets_train = (
        targets_train.cat.codes
    )  # This converts category codes to integers but keeps it as a pandas Series
    targets_train = np.array(
        targets_train
    )  # Convert to numpy array for flexible data manipulation
    for i in range(len(targets_train)):
        if random.random() < args.corruption_rate:
            new_label = random.randint(0, numClasses - 1)
            while new_label == targets_train[i]:
                new_label = random.randint(0, numClasses - 1)
            targets_train[i] = new_label

    # targets_train_corrupted = pd.Categorical(targets_train_int, categories=targets_train.cat.categories)
    knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=args.k_neighbours)

    knn.fit(data_train, targets_train)

    prediction: np.ndarray = knn.predict(data_test)

    targets_test = targets_test.astype(int)  # Convert test labels to int
    prediction = prediction.astype(int)  # Ensure predictions are also int, if necessary

    accuracy: float = accuracy_score(targets_test, prediction)
    conf_matrix = confusion_matrix(targets_test, prediction)

    sensitivity_arr: List[float] = []
    specificity_arr: List[float] = []

    for i in range(numClasses):
        TP: int = conf_matrix[i][i]
        FN: int = np.sum(conf_matrix[i, :]) - TP
        FP: int = np.sum(conf_matrix[:, i]) - TP
        TN: int = np.sum(conf_matrix) - TP - FN - FP

        sensitivity: int = TP / (TP + FN)
        specificity: int = TN / (TN + FP)

        sensitivity_arr.append(sensitivity)
        specificity_arr.append(specificity)

    avg_sens: float = round(np.sum(sensitivity_arr) / numClasses, 4) * 100
    avg_spec: float = round(np.sum(specificity_arr) / numClasses, 4) * 100

    with open(f"{args.base_folder}/csv_data/data.csv", "a", newline="") as file:
        data_writer: csv.Writer = csv.writer(file)

        data_writer.writerow(
            [
                float(args.corruption_rate),
                round(float(accuracy), 4) * 100,
                round(float(avg_sens), 4),
                round(float(avg_spec), 4),
            ]
        )

    timestamp: str = time.strftime("%Y%m%d_%H%M%S")

    job_metadata: dict = {
        "creation_date": timestamp,
        "dataset": args.dataset,
        "num_classes": numClasses,
        "corruption_arr": args.all_corruption,
        "data_chunk": args.data_chunk,
        "k_neighbours": args.k_neighbours,
    }

    with open(f"{args.base_folder}/job_details.json", "w") as file:
        json.dump(job_metadata, file, indent=4)


if __name__ == "__main__":
    main()
