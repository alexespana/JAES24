"""
This file contains utility functions that are used in the backend.

Functions:
is_repository_url(url: str) -> bool
is_a_known_repository(url: str) -> bool
"""
import re
import pandas as pd
from constants import STATIC_FOLDER, REPOSITORIES_CSV_FILE

def is_repository_url(url):
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

def get_owner(url):
    """
    Get the owner of the repository from the repository URL.

    Args:
    url (str): The URL of the repository.

    Returns:
    str: The owner of the repository.
    """
    owner = url.split('/')[-2]
    return owner

def get_repo_name(url):
    """
    Get the name of the repository from the repository URL.

    Args:
    url (str): The URL of the repository.
    
    Returns:
    str: The name of the repository.
    """
    repo_name = url.split('/')[-1]
    return repo_name

def replace_fields(url, owner=None, repo_name=None, pull_number=None, run_id=None):
    url = url.replace('OWNER', owner).replace('REPO', repo_name).replace('PULL_NUMBER', pull_number).replace('RUN_ID', run_id)
    return url
