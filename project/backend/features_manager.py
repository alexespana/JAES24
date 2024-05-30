"""
This file contains utility functions that extract useful information
from the Github Actions metadata. This information (features) is used to
train the model and make predictions later on.
"""
import re
import datetime
import pandas as pd
from constants import HOUR_CONVERTER, FEATURES_FOLDER
from github_manager import GithubManager
from typing import Tuple
from utils import get_owner, get_repo_name

# Indicate type annotations
def get_performance_short(run_id: int, builds: dict) -> float:
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
    found_target_build = False
    builds_after_target = []

    for run in builds["workflow_runs"]:
        if found_target_build:
            builds_after_target.append(run)
            if len(builds_after_target) <= threshold:
                if run["conclusion"] == "success":
                    successful_builds += 1
            else:
                break

        if run["id"] == run_id:
            found_target_build = True


    return (successful_builds / max(len(builds_after_target), 1)) * 100 

def get_performance_long(run_id: int, builds: dict) -> float:
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
    found_target_build = False
    builds_after_target = 0

    for run in builds["workflow_runs"]:
        if found_target_build:
            builds_after_target += 1
            if run["conclusion"] == "success":
                successful_builds += 1
           
        if run["id"] == run_id:
            found_target_build = True

    return (successful_builds / max(builds_after_target, 1)) * 100 

def get_time_frequency(run_id: int, builds: dict) -> int:
    """
    Calculate the time frequency between the target build and the previous build.

    Args:
    run_id (int): The run_id of the build to search for.
    builds (dict): The dictionary containing the builds metadata.

    Returns:
    float: The time frequency between the target build and the previous build (hours).
    """
    previous_build_search = False
    created_target_build = None
    created_previous_build = None


    for run in builds["workflow_runs"]:
        if previous_build_search:
            created_previous_build = run['created_at']
            previous_build_search = False

        if run["id"] == run_id:
            created_target_build = run['created_at']
            previous_build_search = True

    created_previous_build = created_previous_build if created_previous_build is not None else created_target_build
    target_date = datetime.datetime.strptime(created_target_build, "%Y-%m-%dT%H:%M:%SZ")
    previous_date = datetime.datetime.strptime(created_previous_build, "%Y-%m-%dT%H:%M:%SZ")

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

def get_num_commits(build: dict) -> int:
    """
    Calculate the number of commits in a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The number of commits in the build.
    """
    pr_number = get_build_pr_number(build)
    full_name = build['repository']['full_name']
    owner, repo_name = full_name.split('/')

    if pr_number is None:
        return 0

    # Get the commits for this PR
    github_manager = GithubManager()
    commits = github_manager.get_pull_request_commits(owner, repo_name, pr_number, number_of_commits=100)

    return len(commits)

def get_num_files_changed(build: dict) -> Tuple[int, int, int, int]:
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
    pr_number = get_build_pr_number(build)
    full_name = build['repository']['full_name']
    owner, repo_name = full_name.split('/')

    if pr_number is None:
        return 0, 0, 0, 0

    # Get the files for this PR
    github_manager = GithubManager()
    files = github_manager.get_pull_request_files(owner, repo_name, pr_number, number_of_files=100)

    files_added = 0
    files_modified = 0
    files_removed = 0
    for file in files:
        if file['status'] == 'added':
            files_added += 1
        elif file['status'] == 'modified':
            files_modified += 1
        elif file['status'] == 'removed':
            files_removed += 1

    return len(files), files_added, files_modified, files_removed

def get_num_lines_changed(build: dict) -> Tuple[int, int, int, int, int]:
    """
    Calculate the number of lines changed, added and removed in a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The number of lines changed in the build.
    int: The number of lines added in the build.
    int: The number of lines removed in the build.
    """
    pr_number = get_build_pr_number(build)
    full_name = build['repository']['full_name']
    owner, repo_name = full_name.split('/')
    lines_added = 0
    lines_removed = 0
    test_lines_changed = 0
    source_code_modified = False
    extensions = [".py", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".java", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".go", ".rs", ".scala", ".hs"]

    if pr_number is None:
        return 0, 0, 0, 0, 0

    # Get the files for this PR
    github_manager = GithubManager()
    files = github_manager.get_pull_request_files(owner, repo_name, pr_number, number_of_files=100)

    # 0: source code modified and no tests modified
    # 1: source code modified and tests modified
    # 2: no source code modified, not applicable
    unit_tests = 2
    for file in files:
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

def get_failure_distance(run_id: int, builds: dict) -> int:
    """
    Search for the run_id build and count the number of successful builds
    until the first failed build.

    Args:
    run_id (int): The run_id of the build to search for.
    builds (dict): The dictionary containing the builds metadata.

    Returns:
    int: The number of successful builds until the first failed build.
    """
    found_target_build = False
    successful_builds = 0

    for run in builds["workflow_runs"]:
        if found_target_build:
            if run["conclusion"] == "failure":
                break
            else:   # success, cancelled, skipped, None
                successful_builds += 1
            
        if run["id"] == run_id:
            found_target_build = True

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

def get_outcome(build: dict) -> str:
    """
    Extract the outcome of the build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    str: The outcome of the build.
    """
    return build['conclusion']

def get_features(repository_url: str, branch: str, csv_file: str) -> None:
    """
    Extract the following features for a specific repository and branch and save them in a csv file.
    Features:
        PS: performance short
        PL: performance long
        TF: time frequency
        NC: number of commits
        FC: number of files changed
        FA: number of files added
        FM: number of files modified
        FR: number of files removed
        LC: number of lines changed
        LA: number of lines added
        LR: number of lines removed
        LT: number of tests lines changed
        UT: a number indicating whether tests have been written
        FD: failure distance
        WD: week day
        DH: day hour
    Class:
        outcome: success or failure

    Args:
    repository_url (str): The URL of the repository.
    branch (str): The branch of the repository.
    csv_file (str): The name of the csv file to save the features.
    """
    owner = get_owner(repository_url)
    repo_name = get_repo_name(repository_url)

    # Get CI builds to train the model
    github_manager = GithubManager()
    builds = github_manager.get_builds(owner=owner, repo_name=repo_name, branch=branch, number_of_builds=20)

    df = pd.DataFrame(columns=['PS', 'PL', 'TF', 'NC', 'FC', 'FA', 'FM', 'FR', 'LC', 'LA', 'LR', 'LT', 'UT' ,'FD', 'WD', 'DH', 'outcome'])

    # Get the data to train the model
    for build in builds["workflow_runs"]:
        build_id = build['id']
        PS = get_performance_short(build_id, builds)
        PL = get_performance_long(build_id, builds)
        TF = get_time_frequency(build_id, builds)
        NC = get_num_commits(build)
        FC, FA, FM, FR = get_num_files_changed(build)
        LC, LA, LR, LT, UT = get_num_lines_changed(build)
        FD = get_failure_distance(build_id, builds)
        WD = get_weekday(build)
        DH = get_hour(build)
        outcome = get_outcome(build)

        # Add CI build
        df.loc[len(df.index)] = [PS, PL, TF, NC, FC, FA, FM, FR, LC, LA, LR, LT, UT, FD, WD, DH, outcome]

    df.to_csv(FEATURES_FOLDER + csv_file, index=False)
