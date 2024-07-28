"""
This file contains utility functions related to Artificial Intelligence (AI) models.
"""
import os
import pickle
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.svm import SVC
from constants import AIMODELS_FOLDER
from utils import print_model_metrics, ndarray_to_dataframe
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def train_all_models(data: pd.DataFrame, pickle_pattern: str, seed = 42) -> Tuple[StandardScaler, MinMaxScaler]:
    """
    Train all models and save them into the specified folder.

    Args:
    data: features and target values
    pickle_pattern: the base pattern to save the models
    seed: the seed for the random number generator

    Returns:
    zscore: the zscore scaler
    minmax: the minmax scaler
    """
    x_train = data
    y_train = data.pop('outcome')

    # Normalize the data
    zscore = StandardScaler()
    minmax = MinMaxScaler()
    x_train_zscore = zscore.fit_transform(x_train)
    x_train_min_max = minmax.fit_transform(x_train)
    x_train_zscore = ndarray_to_dataframe(x_train.columns, x_train_zscore)
    x_train_min_max = ndarray_to_dataframe(x_train.columns, x_train_min_max)

    dt = DecisionTreeClassifier(random_state=seed)
    train(dt, x_train, y_train)
    with open(get_model_path(pickle_pattern, DT_CLASSIFIER), 'wb') as file:
        pickle.dump(dt, file)

    rf = RandomForestClassifier(random_state=seed)
    train(rf, x_train, y_train)
    with open(get_model_path(pickle_pattern, RF_CLASSIFIER), 'wb') as file:
        pickle.dump(rf, file)

    lr = LogisticRegression(random_state=seed, max_iter=15000)
    train(lr, x_train_zscore, y_train)
    with open(get_model_path(pickle_pattern, LR_CLASSIFIER), 'wb') as file:
        pickle.dump(lr, file)

    svc = SVC(random_state=seed, probability=True)
    train(svc, x_train_zscore, y_train)
    with open(get_model_path(pickle_pattern, SVM_CLASSIFIER), 'wb') as file:
        pickle.dump(svc, file)

    knn = KNeighborsClassifier()
    train(knn, x_train_min_max, y_train)
    with open(get_model_path(pickle_pattern, KNN_CLASSIFIER), 'wb') as file:
        pickle.dump(knn, file)

    nn = MLPClassifier(random_state=seed, max_iter=15000)
    train(nn, x_train_min_max, y_train)
    with open(get_model_path(pickle_pattern, NN_CLASSIFIER), 'wb') as file:
        pickle.dump(nn, file)


    return zscore, minmax

def train_and_evaluate_all_models(data: pd.DataFrame, pickle_pattern: str, test_percentage: float = 0.2, with_accumulation: bool = False,  seed: int = 42):
    """
    Train and evaluate all models for a given dataset.

    Args:
    data: features and target values
    pickle_pattern: the base pattern to save the models
    test_percentage: the percentage of the data to be used for testing
    with_accumulation: whether to accumulate the predictions
    seed: the seed for the random number generator

    Returns:
    None
    """
    # Total number of builds
    total_rows = len(data)

    # Number of builds for training and testing
    train_size = int(total_rows * (1 - test_percentage))
    test_size = total_rows - train_size

    # Ratio of successful and failed builds
    value_counts = data['outcome'].value_counts()
    failure_proportion = value_counts[0] / total_rows
    success_proportion = value_counts[1] / total_rows

    # TRAIN: The first (1 - {test_percentage}) of builds (older)
    x_train = data[:train_size]

    # TEST: The last {test_percentage} of builds (latest)
    x_test = data[-test_size:]

    # True targets for the test set 
    y_test = x_test.pop('outcome')

    # Train models and get transformers
    zscore, minmax = train_all_models(x_train, pickle_pattern, seed)

    # TESTING
    predictions_dt, predictions_prob_dt = predict(get_model_path(pickle_pattern, DT_CLASSIFIER), x_test, y_test, with_accumulation)
    plot_confusion_matrix(y_test, predictions_dt, pickle_pattern, DT_CLASSIFIER)
    plot_roc_curve(y_test, predictions_prob_dt, pickle_pattern, DT_CLASSIFIER)

    predictions_rf, predictions_prob_rf = predict(get_model_path(pickle_pattern, RF_CLASSIFIER), x_test, y_test, with_accumulation)
    plot_confusion_matrix(y_test, predictions_rf, pickle_pattern, RF_CLASSIFIER)
    plot_roc_curve(y_test, predictions_prob_rf, pickle_pattern, RF_CLASSIFIER)

    predictions_lr, predictions_prob_lr = predict(get_model_path(pickle_pattern, LR_CLASSIFIER), x_test, y_test, with_accumulation, zscore)
    plot_confusion_matrix(y_test, predictions_lr, pickle_pattern, LR_CLASSIFIER)
    plot_roc_curve(y_test, predictions_prob_lr, pickle_pattern, LR_CLASSIFIER)

    predictions_svm, predictions_prob_svm = predict(get_model_path(pickle_pattern, SVM_CLASSIFIER), x_test, y_test, with_accumulation, zscore)
    plot_confusion_matrix(y_test, predictions_svm, pickle_pattern, SVM_CLASSIFIER)
    plot_roc_curve(y_test, predictions_prob_svm, pickle_pattern, SVM_CLASSIFIER)

    predictions_knn, predictions_prob_knn = predict(get_model_path(pickle_pattern, KNN_CLASSIFIER), x_test, y_test, with_accumulation, minmax)
    plot_confusion_matrix(y_test, predictions_knn, pickle_pattern, KNN_CLASSIFIER)
    plot_roc_curve(y_test, predictions_prob_knn, pickle_pattern, KNN_CLASSIFIER)

    predictions_nn, predictions_prob_nn = predict(get_model_path(pickle_pattern, NN_CLASSIFIER), x_test, y_test, with_accumulation, minmax)
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
        file.write('- Ratio of pass/fail builds = ' + str(success_proportion) + '/' + str(failure_proportion) + '\n')

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

def k_fold_cross_validation(data: pd.DataFrame, pickle_pattern: str, k: int = 11, with_accumulation: bool = False, seed: int = 42):
    """
    Perform k-fold cross validation build features.

    Args:
    data: features and target values
    pickle_pattern: the base pattern to save the models
    k: the number of folds
    seed: the seed for the random number generator

    Returns:
    None
    """
    k_fold_size = len(data) // k


    shutil.rmtree(AIMODELS_FOLDER + pickle_pattern + '/k-Fold Cross-Validation', ignore_errors=True)  
    os.makedirs(AIMODELS_FOLDER + pickle_pattern + '/k-Fold Cross-Validation', exist_ok=True)

    for i in range(1, k):
        fold_folder = AIMODELS_FOLDER + pickle_pattern + '/k-Fold Cross-Validation/Fold-' + str(i)
        os.makedirs(fold_folder, exist_ok=True)

        train_end_index = i * k_fold_size

        # Train susbet
        train_data = data[:train_end_index]

        # Test subset
        x_test = data[train_end_index + 1:train_end_index + k_fold_size]

        # True targets for the test set 
        y_test = x_test.pop('outcome')

        # TRAIN
        zscore, minmax = train_all_models(train_data, pickle_pattern, seed)

        # TEST
        predictions_dt, predictions_prob_dt = predict(get_model_path(pickle_pattern, DT_CLASSIFIER), x_test, y_test, with_accumulation)
        plot_confusion_matrix(y_test, predictions_dt, pickle_pattern, DT_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_dt, pickle_pattern, DT_CLASSIFIER)

        predictions_rf, predictions_prob_rf = predict(get_model_path(pickle_pattern, RF_CLASSIFIER), x_test, y_test, with_accumulation)
        plot_confusion_matrix(y_test, predictions_rf, pickle_pattern, RF_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_rf, pickle_pattern, RF_CLASSIFIER)

        predictions_lr, predictions_prob_lr = predict(get_model_path(pickle_pattern, LR_CLASSIFIER), x_test, y_test, with_accumulation, zscore)
        plot_confusion_matrix(y_test, predictions_lr, pickle_pattern, LR_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_lr, pickle_pattern, LR_CLASSIFIER)

        predictions_svm, predictions_prob_svm = predict(get_model_path(pickle_pattern, SVM_CLASSIFIER), x_test, y_test, with_accumulation, zscore)
        plot_confusion_matrix(y_test, predictions_svm, pickle_pattern, SVM_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_svm, pickle_pattern, SVM_CLASSIFIER)

        predictions_knn, predictions_prob_knn = predict(get_model_path(pickle_pattern, KNN_CLASSIFIER), x_test, y_test, with_accumulation, minmax)
        plot_confusion_matrix(y_test, predictions_knn, pickle_pattern, KNN_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_knn, pickle_pattern, KNN_CLASSIFIER)

        predictions_nn, predictions_prob_nn = predict(get_model_path(pickle_pattern, NN_CLASSIFIER), x_test, y_test, with_accumulation, minmax)
        plot_confusion_matrix(y_test, predictions_nn, pickle_pattern, NN_CLASSIFIER)
        plot_roc_curve(y_test, predictions_prob_nn, pickle_pattern, NN_CLASSIFIER)

        # Save the evaluation in a file
        with open(AIMODELS_FOLDER + pickle_pattern + '/' + pickle_pattern + '_evaluation.txt', 'w') as file:
            file.write('=================================================================\n')
            file.write('\t~~~~~~~ ' + str(i) +'-Fold Cross Validation for ' + pickle_pattern + ' ~~~~~~~')
            file.write('\n=================================================================\n')
            file.write('- Total number of builds = ' + str(train_end_index + k_fold_size) + '\n')
            file.write('- Training size = ' + str(len(train_data)) + '\n')
            file.write('- Test size = ' + str(len(x_test)) + '\n')

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

        # Move all results for this k-fold to the fold folder
        for item in os.listdir(AIMODELS_FOLDER + pickle_pattern):
            src_path = os.path.join(AIMODELS_FOLDER + pickle_pattern, item)
            dest_path = os.path.join(fold_folder, item)

            if os.path.isfile(src_path):
                shutil.move(src_path, dest_path)

def predict(model_path, x_test, y_test, with_accumulation, transformer = None) -> Tuple[list, list]:
    """
    Make predictions.

    Args:
    model_path (str): The path of the trained model.
    x_test (DataFrame): The test data.
    y_test (DataFrame): The actual target values.
    with_accumulation (bool): Whether to accumulate the predictions.
    transformer: The transformer to normalize the data.

    Returns:
    predictions: The predicted target values.
    predictions_prob: The predicted probabilities.
    """
    with open(model_path,'rb') as file:
        model = pickle.load(file)

    if not with_accumulation:
        # Normalize the data
        if transformer is not None:
            columns = x_test.columns
            x_test = transformer.transform(x_test)
            x_test = ndarray_to_dataframe(columns, x_test)

        # Make predictions for all builds directly
        predictions = model.predict(x_test)
        predictions_prob = model.predict_proba(x_test)[:, 1]
    else:
        # Cumulative predictions
        last_prediction = None
        last_features = None
        in_failure_sequence = False
        predictions = np.zeros(len(x_test))
        predictions_prob = np.zeros(len(x_test))

        # Create an empty DataFrame
        build_features = pd.DataFrame(columns=x_test.columns)
        build_features.loc[0] = [None] * len(x_test.columns)

        for index, (_, row) in enumerate(x_test.iterrows()):
            if in_failure_sequence:     # Subsequent failures
                # Predict automatically a failure
                predictions[index] = 0
                predictions_prob[index] = 0
                last_prediction = 0

                # Case: False positive
                if y_test.iloc[index] == 1:
                    in_failure_sequence = False
            else:
                build_features.iloc[0] = row

                if last_prediction == 1: 
                    build_features =  accumulate_features(last_features, build_features)

                # Save the last features before normalization
                last_features = build_features.copy()

                # Normalize the data
                if transformer is not None:
                    build_features = transformer.transform(build_features)
                    build_features = ndarray_to_dataframe(x_test.columns, build_features)
        
                prediction_i = model.predict(build_features)
                prediction_prob_i = model.predict_proba(build_features)[:, 1]

                predictions[index] = prediction_i
                predictions_prob[index] = prediction_prob_i

                last_prediction = prediction_i

                if last_prediction == 0:    # Check if the last prediction was a failure
                    last_features.loc[0] = [None] * len(x_test.columns)
                    if y_test.iloc[index] == 0:     # First failure
                        in_failure_sequence = True

    return predictions, predictions_prob

def accumulate_features(old_features: pd.DataFrame, new_feature: pd.DataFrame) -> pd.DataFrame:
    """
    Accumulate build changes.

    Args:
    old_features (DataFrame): first build features.
    new_feature (DataFrame): second build features.

    Returns:
    DataFrame: The accumulated features.
    """
    result  = new_feature.copy()

    if 'NC' in result.columns:
        result['NC'] = old_features['NC'] + new_feature['NC']

    if 'FC' in result.columns:
        result['FC'] = old_features['FC'] + new_feature['FC']

    if 'FA' in result.columns:
        result['FA'] = old_features['FA'] + new_feature['FA']

    if 'FM' in result.columns:
        result['FM'] = old_features['FM'] + new_feature['FM']

    if 'FR' in result.columns:
        result['FR'] = old_features['FR'] + new_feature['FR']

    if 'LC' in result.columns:  
        result['LC'] = old_features['LC'] + new_feature['LC']

    if 'LA' in result.columns:
        result['LA'] = old_features['LA'] + new_feature['LA']

    if 'LR' in result.columns:
        result['LR'] = old_features['LR'] + new_feature['LR']

    if 'LT' in result.columns:
        result['LT'] = old_features['LT'] + new_feature['LT']

    return result

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
    precision = precision_score(y_test, predictions, pos_label=0, zero_division=0)
    recall = recall_score(y_test, predictions, pos_label=0, zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label=0, zero_division=0)
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
    plt.close('all')

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
    plt.close('all')
