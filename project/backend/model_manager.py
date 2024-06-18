"""
This file contains utility functions related to Artificial Intelligence (AI) models.
"""
import os
import pickle
import pandas as pd
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

def train_all(train_data, pickle_pattern, model_evaluation = False, test_percentage=0.2, seed = 42):
    """
    Train all models.

    Args:
    x_train (DataFrame): The training data.
    y_train (DataFrame): The target values.

    Returns:
    dict: The trained models.
    """
    if model_evaluation:       
        total_rows = len(train_data)
        train_size = int(total_rows * (1 - test_percentage))
        test_size = total_rows - train_size
        
        # Split the data into training and test sets
        # Training
        x_train = train_data[:train_size]
        y_train = x_train.pop('outcome')

        # Test
        x_test = train_data[-test_size:]
        y_test = x_test.pop('outcome')
    else:
        x_train = train_data
        y_train = train_data.pop('outcome')

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


    # Save the models evaluation in a file
    if model_evaluation:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
    
        predictions_dt = predict(get_model_path(pickle_pattern, 'Decision Tree'), x_test)
        accurary_dt = predictions_dt == y_test

        predictions_rf = predict(get_model_path(pickle_pattern, 'Random Forest'), x_test)
        accurary_rf = predictions_rf == y_test

        predictions_lr = predict(get_model_path(pickle_pattern, 'Linear Regression'), x_test)
        accurary_lr = predictions_lr == y_test

        predictions_svm = predict(get_model_path(pickle_pattern, 'Support Vector Machine'), x_test)
        accurary_svm = predictions_svm == y_test

        predictions_knn = predict(get_model_path(pickle_pattern, 'K-Nearest Neighbors'), x_test)
        accurary_knn = predictions_knn == y_test

        predictions_nn = predict(get_model_path(pickle_pattern, 'Neural Network'), x_test)
        accurary_nn = predictions_nn == y_test

        # Save the evaluation in a file
        with open(AIMODELS_FOLDER + pickle_pattern + '/' + pickle_pattern + '_evaluation.txt', 'w') as file:
            file.write('================ Evaluation of the models for ' + pickle_pattern + ' ================\n')
            file.write('Total builds ==> ' + str(total_rows) + '\n')
            file.write('Training size ==> ' + str(train_size) + '\n')
            file.write('Test size ==> ' + str(test_size) + '\n')
            file.write('Decision Tree ==> ' + str(accurary_dt.mean()) + '\n')
            file.write('\n')
            file.write('Random Forest ==> ' + str(accurary_rf.mean()) + '\n')
            file.write('\n')
            file.write('Linear Regression ==> ' + str(accurary_lr.mean()) + '\n')
            file.write('\n')
            file.write('Support Vector Machine ==> ' + str(accurary_svm.mean()) + '\n')
            file.write('\n')
            file.write('K-Nearest Neighbors ==> ' + str(accurary_knn.mean()) + '\n')
            file.write('\n')
            file.write('Neural Network ==> ' + str(accurary_nn.mean()) + '\n')

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
