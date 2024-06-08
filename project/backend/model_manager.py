"""
This file contains utility functions related to Artificial Intelligence (AI) models.
"""
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from constants import AIMODELS_FOLDER

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

def train(model, x_train, y_train):
    """
    Train the model.

    Args:
    model (object): The instance of the classifier.
    x_train (DataFrame): The training data.
    y_train (DataFrame): The target values.

    Returns:
    object: The trained model.
    """
    model.fit(x_train, y_train)

    return model

def train_all(x_train, y_train, pickle_pattern, seed=42):
    """
    Train all models.

    Args:
    x_train (DataFrame): The training data.
    y_train (DataFrame): The target values.

    Returns:
    dict: The trained models.
    """
    dt = DecisionTreeClassifier(random_state=seed)
    train(dt, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Decision Tree'), 'wb') as file:
        pickle.dump(dt, file)

    rf = RandomForestClassifier(random_state=seed)
    train(rf, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Random Forest'), 'wb') as file:
        pickle.dump(dt, file)

    lr = LinearRegression()
    train(lr, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Linear Regression'), 'wb') as file:
        pickle.dump(dt, file)

    svc = SVC(random_state=seed)
    train(svc, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Support Vector Machine'), 'wb') as file:
        pickle.dump(dt, file)

    knn = KNeighborsClassifier()
    train(knn, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'K-Nearest Neighbors'), 'wb') as file:
        pickle.dump(dt, file)

    nn = MLPClassifier(random_state=seed)
    train(nn, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Neural Network'), 'wb') as file:
        pickle.dump(dt, file)
