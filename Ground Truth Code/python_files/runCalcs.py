from functions import fileUtils
import os
import argparse
from typing import List
from enums.metricEnum import MetricType
import pandas as pd
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="runCalcs Parser")
    parser.add_argument("--jobid", type=int, help="This job's id", required=True)
    args = parser.parse_args()
    thisJob: str = "files_" + str(args.jobid)
    thisJobDir = os.path.join("../job_outputs", thisJob)

    thisJobsMetadata: dict = fileUtils.getJobMetadata(thisJobDir)

    allJobs: List[str] = fileUtils.getAllJobs("../job_outputs")

    fileUtils.setVersion(allJobs, thisJob)
    # Get the updated metadata w/ versioning
    thisJobsMetadata = fileUtils.getJobMetadata(thisJobDir)
    matchingJobs: List[str] = fileUtils.getMatchingJobs(thisJobsMetadata, allJobs)

    for metric in MetricType:
        fileUtils.computeSummaryStats(matchingJobs, thisJob, metric.value)

    fileUtils.computeRobustness(thisJob)


if __name__ == "__main__":
    main()
