"""
This file contains utility functions that extract useful information
from the Github Actions metadata. This information (features) is used to
train the model and make predictions later on.
"""
import datetime
from constants import HOUR_CONVERTER

def get_performance_short(run_id, builds):
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


    return (successful_builds / min(len(builds_after_target), threshold)) * 100 

def get_performance_long(run_id, builds):
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

def get_time_frequency(run_id, builds):
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

    target_date = datetime.datetime.strptime(created_target_build, "%Y-%m-%dT%H:%M:%SZ")
    previous_date = datetime.datetime.strptime(created_previous_build, "%Y-%m-%dT%H:%M:%SZ")

    return round((target_date - previous_date).total_seconds() / HOUR_CONVERTER)
    

    
def get_num_commits(build):
    pass

def get_num_files_changed(build):
    pass

def get_num_lines_changed(build):
    pass

def get_num_tests_lines_changed(build):
    pass

def get_failure_distance(run_id, builds):
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


def get_weekday(build):
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

def get_hour(build):
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