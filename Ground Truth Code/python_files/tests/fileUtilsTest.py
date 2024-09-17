from pathlib import Path
import sys

# Calculate the path to the root of the project
# This assumes that `python_files` is a subdirectory in your project's root
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_root)

from enums.metricEnum import MetricType
from functions import fileUtils
import unittest
from unittest.mock import patch
from pathlib import Path
import os
from os.path import join as real_join
import pandas as pd
from typing import List, Set

currentDir: str = os.getcwd()


# Calling this file from the python_files directory
mockJobsDir = os.path.join(currentDir, "tests/mock_data/mock_jobs")
testFileRegular = os.path.join(mockJobsDir, "testFileRegular")
testFileNoData = os.path.join(mockJobsDir, "testFileNoData")
testFileFull = os.path.join(mockJobsDir, "testFileFull")

# Copy of testFileRegular's data.csv
mockData = {
    "corruption_rate": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "epoch": [1, 2, 3, 4, 5, 6],
    "accuracy": [97.39, 95.44, 99.27, 99.4, 99.47, 99.45],
    "training_loss": [15.94, 3.31, 2.32, 1.78, 1.5, 1.22],
    "test_loss": [0.08, 0.13, 0.02, 0.02, 0.02, 0.02],
    "avg_sens": [97.36, 95.3, 99.26, 99.39, 99.47, 99.44],
    "avg_spec": [99.71, 99.49, 99.92, 99.92999999999999, 99.94, 99.94],
}
regularDataframe: pd.DataFrame = pd.DataFrame(mockData)

# Copy of testfileFull's metadata.csv
mockMetadata = {
    "num_epochs": [12] * 19,
    "seed_value": [1] * 19,
    "min_size": [1] * 19,
    "corruption_rates": [
        0,
        0.25,
        0.5,
        0.6,
        0.7,
        0.75,
        0.775,
        0.7875,
        0.8,
        0.8125,
        0.825,
        0.8375,
        0.85,
        0.8625,
        0.875,
        0.8875,
        0.9,
        0.95,
        1,
    ],
    "num_classes": [6] * 19,
    "dataset": ["FashionMNIST"] * 19,
    "model_architecture": ["CNN"] * 19,
}


matchingSummaryJobs: List[str] = ["files_43066789", "files_43064676"]

files_42782841: str = os.path.join(mockJobsDir, "files_42782841")
files_42782842: str = os.path.join(mockJobsDir, "files_42782842")
files_42782843: str = os.path.join(mockJobsDir, "files_42782843")
files_43064676: str = os.path.join(mockJobsDir, "files_43064676")
files_43066789: str = os.path.join(mockJobsDir, "files_43066789")
files_getJobMetadata: str = os.path.join(mockJobsDir, "files_getJobMetadata")

mockSummaryAcc = {
    "corruption_rate": [0.00],
    "files_43066789": [81.63],
    "files_43064676": [81.63],
    "mean": [81.63],
    "median": [81.63],
    "stdev": [0],
    "variance": [0],
    "range": [0],
    "q25": [81.63],
    "q75": [81.63],
    "min": [81.63],
    "max": [81.63],
    "IQR": [0],
}

mockSummaryDf = pd.DataFrame(mockSummaryAcc)


mockJobDetails = {
    "model_name": "CNN",
    "creation_date": "20231210_143809",
    "hyperparams": {
        "input_size": 1,
        "conv_size_1": 14,
        "conv_size_2": 28,
        "fc1_size": 128,
        "kernel_size_conv": 3,
        "conv_stride": 1,
        "pool_stride": 2,
        "dropout_1": 0.25,
        "dropout_2": 0.25,
        "kernel_size_pool": 2,
        "output_size": 10,
    },
    "optimizer": "SGD",
    "criterion": "CrossEntropyLoss",
    "scheduler": "StepLR",
    "dataset": "MNIST",
    "num_classes": 10,
    "corruption_arr": ["0", "0.25", "0.5"],
    "num_epochs": 3,
    "data_chunk": 1,
    "num_version": 1,
}


class testFileUtils(unittest.TestCase):
    def assertDataFramesEqual(self, df1, df2):
        try:
            pd.testing.assert_frame_equal(df1, df2, atol=1e-4)
        except AssertionError as e:
            raise self.failureException(e)

    def test_getDataframe_with_data(self) -> None:
        df: pd.DataFrame = fileUtils.getDataframe(testFileRegular, "data.csv")
        self.assertDataFramesEqual(regularDataframe, df)

    def test_getDataframe_without_data(self) -> None:
        df_none: pd.DataFrame = fileUtils.getDataframe(testFileNoData, "data.csv")
        self.assertIsNone(df_none, "Did not return None for no data.csv")

    def test_getJobMetadata(self) -> None:
        jobDetailsJson: dict = fileUtils.getJobMetadata(files_getJobMetadata)
        expectedReturn: dict = mockJobDetails
        self.assertEqual(
            expectedReturn,
            jobDetailsJson,
            "The job_details.json file was not returned successfully",
        )

    @patch("os.path.isdir")
    @patch("os.listdir")
    def test_getAllJobs(self, mock_listdir, mock_isdir) -> None:
        mock_listdir.return_value: List[str] = [
            "files_1",
            "files_2",
            "files_3",
            "numbers.txt",
            "cats.png",
            "randomDir",
        ]
        directories: Set[str] = {"files_1", "files_2", "files_3", "randomDir"}
        mock_isdir.side_effect: List[str] = lambda x: os.path.basename(x) in directories

        returnList: List[str] = fileUtils.getAllJobs("dummy/path")
        expectedReturn: List[str] = ["files_1", "files_2", "files_3"]

        self.assertEqual(
            returnList, expectedReturn, "Did not return correct job directories"
        )

    @patch("os.path.join")
    def test_getMatchingJobs(self, mock_join) -> None:
        # Setup the mock paths for each job
        mock_paths = {
            "files_getJobMetadata": files_getJobMetadata,
            "files_43066789": files_43066789,
            "files_43064676": files_43064676,
        }

        # Custom join decided which version of os.path.join to use, the mock or real, since getMatchingJobs calls the function twice in diff places
        # It is called in the getDataframe function dependency
        # This bug is crazy hard to track remember how to fix it please. Be careful with mocking!!!!
        def custom_join(*args):
            # Check if the second argument is one of the keys in mock_paths
            if args[1] in mock_paths:
                return mock_paths[args[1]]
            # If not, return the real os.path.join result
            return real_join(*args)

        # Use a lambda function for side_effect to return the corresponding mock path
        # The args expansion takes the arguments passed to os.path.join , takes the second arg ie.(../job_outputs, job),
        # args[1] will be the job. If no file is found, returns default_path
        mock_join.side_effect = custom_join

        returnList: List[str] = fileUtils.getMatchingJobs(
            mockJobDetails, ["files_getJobMetadata", "files_43066789", "files_43064676"]
        )
        expectedReturn: List[str] = ["files_getJobMetadata"]

        self.assertEqual(returnList, expectedReturn, "Did not filter for matching jobs")

    @patch("pandas.DataFrame.to_csv")
    @patch("os.path.join")
    @unittest.skip("A little complicated")
    def test_computeSummaryStats(self, mock_join, mock_to_csv) -> None:
        mock_paths = {
            "files_43066789": files_43066789,
            "files_43064676": files_43064676,
        }

        def custom_join(*args):
            # Check if the second argument is one of the keys in mock_paths
            if args[1] in mock_paths:
                return mock_paths[args[1]]
            # If not, return the real os.path.join result
            return real_join(*args)

        mock_join.side_effect = custom_join

        metric = MetricType.ACCURACY.value
        returnJobDf: pd.DataFrame = fileUtils.computeSummaryStats(
            matchingSummaryJobs, "files_43064676", metric
        )
        mock_to_csv.assert_called_once()
        expectedReturn: pd.DataFrame = mockSummaryDf
        self.assertDataFramesEqual(returnJobDf, expectedReturn)


if __name__ == "__main__":
    unittest.main(exit=False)
