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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)
from constants import (
    DT_CLASSIFIER, RF_CLASSIFIER,
    LR_CLASSIFIER, SVM_CLASSIFIER,
    KNN_CLASSIFIER, NN_CLASSIFIER
)

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

    if classifier == DT_CLASSIFIER:
        model_type = 'dt'
    elif classifier == RF_CLASSIFIER:
        model_type = 'rf'
    elif classifier == LR_CLASSIFIER:
        model_type = 'lr'
    elif classifier == SVM_CLASSIFIER:
        model_type = 'svm'
    elif classifier == KNN_CLASSIFIER:
        model_type = 'knn'
    elif classifier == NN_CLASSIFIER:
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
    with open(get_model_path(pickle_pattern, DT_CLASSIFIER), 'wb') as file:
        pickle.dump(dt, file)

    rf = RandomForestClassifier(random_state=seed)
    train(rf, x_train, y_train)
    with open(get_model_path(pickle_pattern, RF_CLASSIFIER), 'wb') as file:
        pickle.dump(rf, file)

    lr = LogisticRegression(random_state=seed, max_iter=15000)
    train(lr, x_train, y_train)
    with open(get_model_path(pickle_pattern, LR_CLASSIFIER), 'wb') as file:
        pickle.dump(lr, file)

    svc = SVC(random_state=seed, probability=True)
    train(svc, x_train, y_train)
    with open(get_model_path(pickle_pattern, SVM_CLASSIFIER), 'wb') as file:
        pickle.dump(svc, file)

    knn = KNeighborsClassifier()
    train(knn, x_train, y_train)
    with open(get_model_path(pickle_pattern, KNN_CLASSIFIER), 'wb') as file:
        pickle.dump(knn, file)

    nn = MLPClassifier(random_state=seed)
    train(nn, x_train, y_train)
    with open(get_model_path(pickle_pattern, NN_CLASSIFIER), 'wb') as file:
        pickle.dump(nn, file)


    # Save the models evaluation in a file
    if model_evaluation:    
        predictions_dt, predictions_prob_dt = predict(get_model_path(pickle_pattern, DT_CLASSIFIER), x_test)
        plot_confusion_matrix(y_test, predictions_dt, pickle_pattern, DT_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_dt, pickle_pattern, DT_CLASSIFIER)

        predictions_rf, predictions_prob_rf = predict(get_model_path(pickle_pattern, RF_CLASSIFIER), x_test)
        plot_confusion_matrix(y_test, predictions_rf, pickle_pattern, RF_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_rf, pickle_pattern, RF_CLASSIFIER)

        predictions_lr, predictions_prob_lr = predict(get_model_path(pickle_pattern, LR_CLASSIFIER), x_test)
        plot_confusion_matrix(y_test, predictions_lr, pickle_pattern, LR_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_lr, pickle_pattern, LR_CLASSIFIER)

        predictions_svm, predictions_prob_svm = predict(get_model_path(pickle_pattern, SVM_CLASSIFIER), x_test)
        plot_confusion_matrix(y_test, predictions_svm, pickle_pattern, SVM_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_svm, pickle_pattern, SVM_CLASSIFIER)

        predictions_knn, predictions_prob_knn = predict(get_model_path(pickle_pattern, KNN_CLASSIFIER), x_test)
        plot_confusion_matrix(y_test, predictions_knn, pickle_pattern, KNN_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_knn, pickle_pattern, KNN_CLASSIFIER)

        predictions_nn, predictions_prob_nn = predict(get_model_path(pickle_pattern, NN_CLASSIFIER), x_test)
        plot_confusion_matrix(y_test, predictions_nn, pickle_pattern, NN_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_nn, pickle_pattern, NN_CLASSIFIER)

        # Save the evaluation in a file
        with open(AIMODELS_FOLDER + pickle_pattern + '/' + pickle_pattern + '_evaluation.txt', 'w') as file:
            file.write('=================================================================\n')
            file.write('\t~~~~~~~ Evaluation of the models for ' + pickle_pattern + ' ~~~~~~~')
            file.write('\n=================================================================\n')
            file.write('- Total number of builds = ' + str(total_rows) + '\n')
            file.write('- Training size = ' + str(train_size) + '\n')
            file.write('- Test size = ' + str(test_size) + '\n')

            file.write(print_model_metrics(DT_CLASSIFIER, *calculate_metrics(y_test, predictions_dt, predictions_prob_dt)))
            file.write('\n')
            file.write(print_model_metrics(RF_CLASSIFIER, *calculate_metrics(y_test, predictions_rf, predictions_prob_rf)))
            file.write('\n')
            file.write(print_model_metrics(LR_CLASSIFIER, *calculate_metrics(y_test, predictions_lr, predictions_prob_lr)))
            file.write('\n')
            file.write(print_model_metrics(SVM_CLASSIFIER, *calculate_metrics(y_test, predictions_svm, predictions_prob_svm)))
            file.write('\n')
            file.write(print_model_metrics(KNN_CLASSIFIER, *calculate_metrics(y_test, predictions_knn, predictions_prob_knn)))
            file.write('\n')
            file.write(print_model_metrics(NN_CLASSIFIER, *calculate_metrics(y_test, predictions_nn, predictions_prob_nn)))
            file.write('\n')

def predict(model_path, x_test) -> Tuple[list, list]:
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
    predictions_prob = model.predict_proba(x_test)[:, 1]
    
    return predictions, predictions_prob

def calculate_metrics(y_test: list, predictions: list, predictions_prob: list)-> Tuple[float, float, float, float, float]:
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
    auc = roc_auc_score(y_test, predictions_prob)


    return cm, acc, precision, recall, f1, auc

def plot_confusion_matrix(y_test: list, predictions: list, pickle_pattern: str, classifier_type: str)-> None:
    """
    Plot the confusion matrix.

    Args:
    y_test (DataFrame): The actual target values.
    predictions (DataFrame): The predicted target values.
    """
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    # Save the plot to a file
    plt.savefig(AIMODELS_FOLDER + pickle_pattern + '/' + 'cm_' + parse_classifier(classifier_type) + '.png')
    plt.close()

def plot_roc_curve(y_test: list, predictions_prob: list, pickle_pattern: str, classifier_type: str)-> None:
    """
    Plot the ROC curve.

    Args:
    y_test (DataFrame): The actual target values.
    predictions_prob (DataFrame): The predicted probabilities.
    """
    roc_auc = roc_auc_score(y_test, predictions_prob)
    fpr, tpr, _ = roc_curve(y_test, predictions_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    # Save the plot to a file
    plt.savefig(AIMODELS_FOLDER + pickle_pattern + '/' + 'roc_' + parse_classifier(classifier_type) + '.png')
    plt.close()
