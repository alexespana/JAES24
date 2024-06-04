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
from constants import FEATURES_FOLDER, AIMODELS_FOLDER
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

def parse_classifier(classifier: str) -> str:
    """
    Parse the classifier name.

    Args:
    classifier (str): The name of the classifier.

    Returns:
    str: The parsed classifier name.
    """
    model_type = ''

    if classifier == 'Decision Tree':
        model_type = 'dt'
    elif classifier == 'Random Forest':
        model_type = 'rf'
    elif classifier == 'Linear Regression':
        model_type = 'lr'
    elif classifier == 'Support Vector Machine':
        model_type = 'svm'
    elif classifier == 'K-Nearest Neighbors':
        model_type = 'knn'
    elif classifier == 'Neural Network':
        model_type = 'nn'

    return model_type

def get_model_path(model_name: str, classifier_type: str) -> str:
    """
    Get the path of the trained model.

    Args:
    model_name (str): The name of the model.

    Returns:
    str: The path of the trained model.
    """
    return AIMODELS_FOLDER + model_name + '_' + parse_classifier(classifier_type) + '.pkl'


def is_model_available(model_name: str, classifier_type: str) -> bool:
    """
    Check if the trained model is available.

    Args:
    model_name (str): The name of the model.

    Returns:
    bool: True if the model is available, False otherwise.
    """
    return os.path.isfile(get_model_path(model_name, classifier_type))

def get_model(classifier_type: str, seed: int = 42):
    """
    Get an instance of the requested classifier.

    Args:
    classifier_type (str): The type of the classifier.

    Returns:
    object: The instance of the classifier.
    """
    model = None

    if classifier_type == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=seed)
    elif classifier_type == 'Random Forest':
        model = RandomForestClassifier(random_state=seed)
    elif classifier_type == 'Linear Regression':
        model = LinearRegression()
    elif classifier_type == 'Support Vector Machine':
        model = SVC(random_state=seed)
    elif classifier_type == 'K-Nearest Neighbors':
        model = KNeighborsClassifier()
    elif classifier_type == 'Neural Network':
        model = MLPClassifier(random_state=seed)
    else:
            raise ValueError(f"Model '{classifier_type}' is not recognized. Please choose "
                             "from 'Decision Tree', 'Random Forest', 'Linear Regression', "
                             "'Support Vector Machine', 'K-Nearest Neighbors', 'Neural Network'.")

    return model
