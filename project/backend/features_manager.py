"""
This file contains utility functions that extract useful information
from the Github Actions metadata. This information (features) is used to
train the model and make predictions later on.
"""
import datetime
from constants import HOUR_CONVERTER
from flask import jsonify
from github_manager import GithubManager

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

def get_num_files_changed(build: dict) -> int:
    """
    Calculate the number of files changed in a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The number of files changed in the build.
    """
    pr_number = get_build_pr_number(build)
    full_name = build['repository']['full_name']
    owner, repo_name = full_name.split('/')

    if pr_number is None:
        return 0

    # Get the files for this PR
    github_manager = GithubManager()
    files = github_manager.get_pull_request_files(owner, repo_name, pr_number, number_of_files=100)

    return len(files)

def get_num_lines_changed(build: dict) -> int:
    """
    Calculate the number of lines changed in a build.

    Args:
    build (dict): The dictionary containing the build metadata.

    Returns:
    int: The number of lines changed in the build.
    """
    pr_number = get_build_pr_number(build)
    full_name = build['repository']['full_name']
    owner, repo_name = full_name.split('/')

    if pr_number is None:
        return 0

    # Get the files for this PR
    github_manager = GithubManager()
    files = github_manager.get_pull_request_files(owner, repo_name, pr_number, number_of_files=100)

    lines_changed = 0
    for file in files:
        lines_changed += file['changes']

    return lines_changed

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
