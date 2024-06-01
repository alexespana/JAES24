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
from constants import FEATURES_FOLDER

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

def replace_fields(url, owner: str = '', repo_name: str = '', pull_number: int = '', run_id: int = '') -> str:
    url = url.replace('OWNER', owner).replace('REPO', repo_name).replace('PULL_NUMBER', pull_number).replace('RUN_ID', run_id)
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