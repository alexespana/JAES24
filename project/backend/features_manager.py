"""
This file contains utility functions that extract useful information
from the Github Actions metadata. This information (features) is used to
train the model and make predictions later on.
"""
import re
import os
import datetime
import time
import pandas as pd
import json
from typing import Tuple
from constants import HOUR_CONVERTER
from concurrent.futures import ThreadPoolExecutor
from github_manager import GithubManager
from utils import get_owner, get_repo_name, get_builds_folder, get_features_folder
from model_manager import train_all

executor = ThreadPoolExecutor(max_workers=4)


# Indicate type annotations
def get_performance_short(run_id: int, builds: dict, id_to_index: dict) -> float:
    """
    This function calculates the performance of a build by counting the number
    of successful builds after the target build.

    Args:
    run_id (int): The run_id of the build to search for.
    builds (dict): The dictionary containing the builds metadata.

    Returns:
    float: The performance of the build in percentage.
    """
    threshold = 5
    successful_builds = 0
    start_index = id_to_index[run_id]
    end_index = start_index + threshold

    successful_builds = sum([1 for run in builds["workflow_runs"][start_index:end_index] if run["conclusion"] == "success"])

    return (successful_builds / max(len(builds["workflow_runs"][start_index:end_index]), 1)) * 100

def get_performance_long(run_id: int, builds: dict, id_to_index: dict) -> float:
    """
    Calculate the performance long term of a repository by counting the number
    of successful builds after the target build.

    Args:
    run_id (int): The run_id of the build to search for.
    builds (dict): The dictionary containing the builds metadata.

    Returns:
    float: The performance of the build in percentage.
    """
    successful_builds = 0
    start_index = id_to_index[run_id]

    successful_builds = sum([1 for run in builds["workflow_runs"][start_index:] if run["conclusion"] == "success"])

    return (successful_builds / max(len(builds["workflow_runs"][start_index:]), 1)) * 100 

def get_time_frequency(run_id: int, builds: dict, id_to_index: dict) -> int:
    """
    Calculate the time frequency between the target build and the previous build.

    Args:
    run_id (int): The run_id of the build to search for.
    builds (dict): The dictionary containing the builds metadata.

    Returns:
    float: The time frequency between the target build and the previous build (hours).
    """
    target_index = id_to_index[run_id]
    previous_index = target_index + 1

    try:
        created_previous_date = builds["workflow_runs"][previous_index]['created_at']
    except IndexError:
        created_previous_date = builds["workflow_runs"][target_index]['created_at']

    target_date = datetime.datetime.strptime(builds["workflow_runs"][target_index]['created_at'], "%Y-%m-%dT%H:%M:%SZ")
    previous_date = datetime.datetime.strptime(created_previous_date, "%Y-%m-%dT%H:%M:%SZ")

    return round((target_date - previous_date).total_seconds() / HOUR_CONVERTER)

def get_build_pr_number(build: dict) -> int:
    """
    Extract the pull request number from a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The pull request number.
    """
    pr_number = None
    # Check if the build is triggered by a pull request event
    if 'pull_request' in build['event']:
        # Get the owner and repository name
        full_name = build['repository']['full_name']
        owner, repo_name = full_name.split('/')
        # Get the name of the PR which trigerred the build
        display_title = build['display_title']
        # Get the label (organization:ref-name) who triggered the build
        label = build['head_repository']['owner']['login'] + ':' + build['head_branch']

        # Get pull requests for this branch
        github_manager = GithubManager()
        pull_requests = github_manager.get_pull_requests(owner, repo_name, label)

        # Search for the PR that triggered the build
        for pr in pull_requests:
            if pr['title'] == display_title:
                pr_number = pr['number']
                break

    return pr_number

def get_num_commits(build: dict, build_pr_number: int) -> int:
    """
    Calculate the number of commits in a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The number of commits in the build.
    """
    full_name = build['repository']['full_name']
    owner, repo_name = full_name.split('/')

    if build_pr_number is None:
        return 0

    # Get the commits for this PR
    github_manager = GithubManager()
    commits = github_manager.get_pull_request_commits(owner, repo_name, build_pr_number, number_of_commits=100)

    return len(commits)

def get_num_files_changed(build: dict, build_pr_number: int, pull_request_files: dict) -> Tuple[int, int, int, int]:
    """
    Calculate the number of files changed, added, modified and removed in a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The number of files changed in the build.
    int: The number of files added in the build.
    int: The number of files modified in the build.
    int: The number of files removed in the build.
    """
    if build_pr_number is None:
        return 0, 0, 0, 0

    files_added = 0
    files_modified = 0
    files_removed = 0
    for file in pull_request_files:
        if file['status'] == 'added':
            files_added += 1
        elif file['status'] == 'modified':
            files_modified += 1
        elif file['status'] == 'removed':
            files_removed += 1

    return len(pull_request_files), files_added, files_modified, files_removed

def get_num_lines_changed(build: dict, build_pr_number: int, pull_request_files: dict) -> Tuple[int, int, int, int, int]:
    """
    Calculate the number of lines changed, added and removed in a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The number of lines changed in the build.
    int: The number of lines added in the build.
    int: The number of lines removed in the build.
    """
    lines_added = 0
    lines_removed = 0
    test_lines_changed = 0
    source_code_modified = False
    extensions = [".py", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".java", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".go", ".rs", ".scala", ".hs"]

    if build_pr_number is None:
        return 0, 0, 0, 0, 0

    # 0: source code modified and no tests modified
    # 1: source code modified and tests modified
    # 2: no source code modified, not applicable
    unit_tests = 2
    for file in pull_request_files:
        lines_added += file['additions']
        lines_removed += file['deletions']

        file_name = file['filename'].lower()
        if 'test' in file_name or 'spec' in file_name:
            test_lines_changed = file['changes']
        else:
            extensions_pattern = "|".join(re.escape(ext) for ext in extensions)
            regex = re.compile(r"\b" + extensions_pattern + r"\b")
            if regex.search(file_name):
                source_code_modified = True

    if source_code_modified:
        if test_lines_changed > 0:
            unit_tests = 1
        else:
            unit_tests = 0

    return lines_added + lines_removed, lines_added, lines_removed, test_lines_changed, unit_tests

def get_failure_distance(run_id: int, builds: dict, id_to_index: dict) -> int:
    """
    Search for the run_id build and count the number of successful builds
    until the first failed build.

    Args:
    run_id (int): The run_id of the build to search for.
    builds (dict): The dictionary containing the builds metadata.

    Returns:
    int: The number of successful builds until the first failed build.
    """
    successful_builds = 0

    start_index = id_to_index[run_id]


    for run in builds["workflow_runs"][start_index + 1:]:
        if run["conclusion"] == "failure":
            break
        
        # success, cancelled, skipped, None
        successful_builds += 1

    return successful_builds


def get_weekday(build: dict) -> int:
    """
    Calculate the weekday of the build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The weekday of the build [0,6].
    """
    created_at = build['created_at']
    date = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")

    return date.weekday()

def get_hour(build: dict) -> int:
    """
    Calculate the hour of the build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The hour of the build [0,23].
    """
    created_at = build['created_at']
    date = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")

    return date.hour

def get_outcome(build: dict) -> int:
    """
    Extract the outcome of the build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    str: The outcome of the build.
    """
    return 0 if build['conclusion'] == 'failure' else 1

def get_builds_by_months(owner: str, repo_name: str, branch: str) -> None:

    start_year = 2019
    current_year = datetime.datetime.now().year

    for year in range(start_year, current_year + 1):
        for month in range(1, 13):
            if year == current_year and month > datetime.datetime.now().month:
                break

            github_manager = GithubManager()
            github_manager.save_month_builds(month, year, owner, repo_name, branch)

def get_features(repo_name: str, branch: str, csv_file: str) -> None:
    """
    Extract all the features from the builds files located in the builds folder
    and save them in a csv file.

    Args:
    repo_name (str): The name of the repository.
    branch (str): The branch of the repository.
    csv_file (str): The name of the csv file to save the features.

    Returns:
    None
    """
    # Get all files from its build folder

    builds_folder = get_builds_folder(repo_name, branch)
    files = os.listdir(builds_folder)

    df = pd.DataFrame(columns=['PS', 'PL', 'TF', 'NC', 'FC', 'FA', 'FM', 'FR', 'LC', 'LA', 'LR', 'LT', 'UT' ,'FD', 'WD', 'DH', 'outcome'])

    github_manager = GithubManager()

    for file in files:
        # Logear el nombre del archivo
        
        with open(builds_folder + file, 'r') as f:
            builds = json.load(f)

        id_to_index = {build['id']: index for index, build in enumerate(builds["workflow_runs"])}

        # Extract the features
        for build in builds["workflow_runs"]:
            build_pr_number = get_build_pr_number(build)
            if build_pr_number is not None:
                full_name = build['repository']['full_name']
                owner, repo_name = full_name.split('/')
                files = github_manager.get_pull_request_files(owner, repo_name, build_pr_number, number_of_files=100)

            build_id = build['id']
            PS = get_performance_short(build_id, builds, id_to_index)
            PL = get_performance_long(build_id, builds, id_to_index)
            TF = get_time_frequency(build_id, builds, id_to_index)
            NC = get_num_commits(build, build_pr_number)
            FC, FA, FM, FR = get_num_files_changed(build, build_pr_number, files)
            LC, LA, LR, LT, UT = get_num_lines_changed(build, build_pr_number, files)
            FD = get_failure_distance(build_id, builds, id_to_index)
            WD = get_weekday(build)
            DH = get_hour(build)
            outcome = get_outcome(build)

            # Add CI build
            df.loc[len(df.index)] = [PS, PL, TF, NC, FC, FA, FM, FR, LC, LA, LR, LT, UT, FD, WD, DH, outcome]

        # Save the features in a csv file or add them to an existing file
        try:
            with open(get_features_folder(repo_name, branch) + csv_file, 'x') as f:
                df.to_csv(f, index=False)
        except FileExistsError:
            df.to_csv(get_features_folder(repo_name, branch) + csv_file, mode='a', header=False, index=False)

        df = df.drop(df.index)

def process_repository(repository_url: str, branch: str, features_file: str, pickle_pattern: str) -> None:
    owner = get_owner(repository_url)
    repo_name = get_repo_name(repository_url)
    
    # Get all the builds from repository from 2019 (GitHub Actions started in 2019) to 2024
    executor.submit(get_builds_by_months, owner, repo_name, branch)

    # Wait some time to collect a sufficient number of builds
    time.sleep(900)     # 15 minutes

    # Extract the features from the builds (located in the builds folder))
    get_features(repo_name, branch, features_file)

    # Train all models with the features extracted
    x_train = pd.read_csv(get_features_folder(repo_name, branch) + features_file)
    y_train = x_train.pop('outcome')
    train_all(x_train, y_train, pickle_pattern)
