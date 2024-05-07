"""
This file contains utility functions that extract useful information
from the Github Actions metadata. This information (features) is used to
train the model and make predictions later on.
"""
import datetime
from constants import HOUR_CONVERTER

def get_performance_short(builds):
    pass

def get_performance_long(builds):
    pass

def get_time_frequency(run_id, builds):
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
    created_at = build['created_at']
    date = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")

    return date.weekday()

def get_hour(build):
    created_at = build['created_at']
    date = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")

    return date.hour