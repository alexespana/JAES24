"""
This file contains utility functions related to Artificial Intelligence (AI) models.
"""
import os
import pickle
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.svm import SVC
from utils import print_model_metrics
from constants import AIMODELS_FOLDER
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def get_model_path(model_name: str, classifier_type: str) -> str:
    """
    Get the path of the trained model.

    Args:
    model_name (str): The name of the model.

    Returns:
    str: The path of the trained model.
    """
    return AIMODELS_FOLDER + model_name + '/' + model_name + '_' + parse_classifier(classifier_type) + '.pkl'

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

def train_all(data, pickle_pattern, model_evaluation = False, test_percentage=0.2, seed = 42):
    """
    Train all models.

    Args:
    x_train (DataFrame): The training data.
    y_train (DataFrame): The target values.

    Returns:
    dict: The trained models.
    """
    if model_evaluation:       
        total_rows = len(data)
        test_size = int(total_rows * test_percentage)
        train_size = total_rows - test_size
        
        # Split the data into training and test sets
        # TRAIN: The last 80% of builds (older)
        x_train = data[-train_size:]
        y_train = x_train.pop('outcome')

        # TEST: The first 20% of builds (latest)
        x_test = data[:test_size]
        y_test = x_test.pop('outcome')
    else:
        x_train = data
        y_train = data.pop('outcome')

    dt = DecisionTreeClassifier(random_state=seed)
    train(dt, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Decision Tree'), 'wb') as file:
        pickle.dump(dt, file)

    rf = RandomForestClassifier(random_state=seed)
    train(rf, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Random Forest'), 'wb') as file:
        pickle.dump(rf, file)

    lr = LinearRegression()
    train(lr, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Linear Regression'), 'wb') as file:
        pickle.dump(lr, file)

    svc = SVC(random_state=seed)
    train(svc, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Support Vector Machine'), 'wb') as file:
        pickle.dump(svc, file)

    knn = KNeighborsClassifier()
    train(knn, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'K-Nearest Neighbors'), 'wb') as file:
        pickle.dump(knn, file)

    nn = MLPClassifier(random_state=seed)
    train(nn, x_train, y_train)
    with open(get_model_path(pickle_pattern, 'Neural Network'), 'wb') as file:
        pickle.dump(nn, file)


    # Save the models evaluation in a file
    if model_evaluation:    
        predictions_dt = predict(get_model_path(pickle_pattern, 'Decision Tree'), x_test)
        plot_confusion_matrix(y_test, predictions_dt, pickle_pattern, 'Decision Tree')

        predictions_rf = predict(get_model_path(pickle_pattern, 'Random Forest'), x_test)
        plot_confusion_matrix(y_test, predictions_rf, pickle_pattern, 'Random Forest')

        predictions_lr = predict(get_model_path(pickle_pattern, 'Linear Regression'), x_test)
        predictions_lr = (predictions_lr > 0.5).astype(int).ravel()
        plot_confusion_matrix(y_test, predictions_lr, pickle_pattern, 'Linear Regression')

        predictions_svm = predict(get_model_path(pickle_pattern, 'Support Vector Machine'), x_test)
        plot_confusion_matrix(y_test, predictions_svm, pickle_pattern, 'Support Vector Machine')

        predictions_knn = predict(get_model_path(pickle_pattern, 'K-Nearest Neighbors'), x_test)
        plot_confusion_matrix(y_test, predictions_knn, pickle_pattern, 'K-Nearest Neighbors')

        predictions_nn = predict(get_model_path(pickle_pattern, 'Neural Network'), x_test)
        plot_confusion_matrix(y_test, predictions_nn, pickle_pattern, 'Neural Network')

        # Save the evaluation in a file
        with open(AIMODELS_FOLDER + pickle_pattern + '/' + pickle_pattern + '_evaluation.txt', 'w') as file:
            file.write('=================================================================\n')
            file.write('\t~~~~~~~ Evaluation of the models for ' + pickle_pattern + ' ~~~~~~~')
            file.write('\n=================================================================\n')
            file.write('- Total number of builds = ' + str(total_rows) + '\n')
            file.write('- Training size = ' + str(train_size) + '\n')
            file.write('- Test size = ' + str(test_size) + '\n')

            file.write(print_model_metrics('Decision Tree', *calculate_metrics(y_test, predictions_dt)))
            file.write('\n')
            file.write(print_model_metrics('Random Forest', *calculate_metrics(y_test, predictions_rf)))
            file.write('\n')
            file.write(print_model_metrics('Linear Regression', *calculate_metrics(y_test, predictions_lr)))
            file.write('\n')
            file.write(print_model_metrics('Support Vector Machine', *calculate_metrics(y_test, predictions_svm)))
            file.write('\n')
            file.write(print_model_metrics('K-Nearest Neighbors', *calculate_metrics(y_test, predictions_knn)))
            file.write('\n')
            file.write(print_model_metrics('Neural Network', *calculate_metrics(y_test, predictions_nn)))
            file.write('\n')

def predict(model_path, x_test):
    """
    Make predictions.

    Args:
    model_path (str): The path of the trained model.
    x_test (DataFrame): The test data.

    Returns:
    DataFrame: The predictions.
    """
    with open(model_path,'rb') as file:
        model = pickle.load(file)

    # Make predictions
    predictions = model.predict(x_test)
    
    return predictions

def calculate_metrics(y_test: list, predictions: list)-> Tuple[float, float, float, float]:
    """
    Calculate the metrics.

    Args:
    y_test (DataFrame): The actual target values.
    predictions (DataFrame): The predicted target values.

    Returns:
    dict: The metrics.
    """
    cm = confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return cm, acc, precision, recall, f1

def plot_confusion_matrix(y_test: list, predictions: list, pickle_pattern: str, classifier_type: str)-> None:
    """
    Plot the confusion matrix.

    Args:
    y_test (DataFrame): The actual target values.
    predictions (DataFrame): The predicted target values.
    """
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    # Save the plot to a file
    plt.savefig(AIMODELS_FOLDER + pickle_pattern + '/' + 'cm_' + parse_classifier(classifier_type) + '.png')
