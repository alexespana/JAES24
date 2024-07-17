"""
This file contains utility functions that are used in the backend.

Functions:
is_repository_url(url: str) -> bool
get_owner(url: str) -> str
get_repo_name(url: str) -> str
replace_fields(url: str, owner: str, repo_name: str, pull_number: str, run_id: str) -> str
"""
import re
import os
import calendar
import numpy as np
import pandas as pd
from typing import Tuple
from constants import FEATURES_FOLDER, AIMODELS_FOLDER

def is_repository_url(url: str) -> bool:
    """
    Check if the given URL is a GitHub repository URL.

    Args:
    url (str): The URL to check.

    Returns:
    bool: True if the URL is a GitHub repository URL, False otherwise.
    """
    result = False
    pattern = r'https?://github\.com/[\w.-]+/[\w.-]+'
    if re.match(pattern, url):
        result = True

    return result

def get_owner(url: str) -> str:
    """
    Get the owner of the repository from the repository URL.

    Args:
    url (str): The URL of the repository.

    Returns:
    str: The owner of the repository.
    """
    owner = url.split('/')[-2]
    return owner

def get_repo_name(url: str) -> str:
    """
    Get the name of the repository from the repository URL.

    Args:
    url (str): The URL of the repository.
    
    Returns:
    str: The name of the repository.
    """
    repo_name = url.split('/')[-1]
    return repo_name

def replace_fields(url, owner: str = '', repo_name: str = '', pull_number: int = '', run_id: int = '', sha: int = '') -> str:
    url = url.replace('OWNER', owner).replace('REPO', repo_name).replace('PULL_NUMBER', pull_number).replace('RUN_ID', run_id).replace('COMMIT_SHA', sha)
    return url

def normalize_branch_name(branch: str) -> str:
    """
    Normalize the branch name.

    Args:
    branch (str): The name of the branch.

    Returns:
    str: The normalized branch name.
    """
    return branch.replace('/', '-')

def is_csv_available(file_name: str) -> bool:
    """
    Check if the CSV file is available.

    Args:
    file_name (str): The name of the file.

    Returns:
    bool: True if the file is available, False otherwise.
    """
    return os.path.isfile(FEATURES_FOLDER + file_name)

def get_month_start_end(year: int, month: int) -> Tuple[str, str]:
    """
    Get the start and end date of the month.

    Args:
    year (int): The year.
    month (int): The month.

    Returns:
    Tuple[str, str]: The start and end date of the month.
    """
    start_date = str(year) + '-' + str(month).zfill(2) + '-01'
    end_date = str(year) + '-' + str(month).zfill(2) + '-' + str(calendar.monthrange(year, month)[1])

    return start_date, end_date

def get_builds_folder(repo_name: str, branch: str) -> str:
    """
    Get the folder path for the builds.

    Args:
    repo_name (str): The name of the repository.
    branch (str): The name of the branch.

    Returns:
    str: The folder path for the builds.
    """
    return FEATURES_FOLDER + repo_name + '_' + normalize_branch_name(branch) + '/builds/'

def get_aimodels_folder(repo_name: str, branch: str) -> str:
    """
    Get the folder path for the AI models.

    Args:
    repo_name (str): The name of the repository.
    branch (str): The name of the branch.

    Returns:
    str: The folder path for the AI models.
    """
    return AIMODELS_FOLDER + repo_name + '_' + normalize_branch_name(branch) + '/'

def get_features_folder(repo_name: str, branch: str) -> str:
    """
    Get the folder path for the CSV files.

    Args:
    repo_name (str): The name of the repository.
    branch (str): The name of the branch.

    Returns:
    str: The folder path for the CSV files.
    """
    return FEATURES_FOLDER + repo_name + '_' + normalize_branch_name(branch) + '/'

def print_model_metrics(model_type: str, confusion_matrix: np.ndarray, acc: float, precision: float, recall: float, f1: float, auc: float) -> str:
    """
    Print the model metrics in a friendly format.

    Args:
    model_type (str): The type of the model.
    confusion_matrix (np.ndarray): The confusion matrix.
    acc (float): The accuracy score.
    precision (float): The precision score.
    recall (float): The recall score.
    f1 (float): The F1 score.
    auc (float): The AUC score.

    Returns:
    str: The model metrics in a friendly format.
    """
    message = "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n" + \
                "Model type: {}\n".format(model_type) + \
                "Confusion matrix:\n" + \
                "{}\n\n".format(confusion_matrix) + \
                "Accuracy: {:.6f}\n".format(acc) + \
                "Precision: {:.6f}\n".format(precision) + \
                "Recall: {:.6f}\n".format(recall) + \
                "F1: {:.6f}\n".format(f1) + \
                "AUC: {:.6f}\n".format(auc) + \
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    
    return message

def ndarray_to_dataframe(columns: list, data: np.ndarray) -> pd.DataFrame:
    """
    Convert a NumPy array to a Pandas DataFrame.

    Args:
    columns (list): The column names.
    data (np.ndarray): The data.

    Returns:
    pd.DataFrame: The Pandas DataFrame.
    """
    return pd.DataFrame(data, columns=columns)
