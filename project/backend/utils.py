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
import glob
import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                "AUC: {:.6f} (-1 = Undefined)\n".format(auc) + \
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    
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

def graph_ratio(features_files, title) -> None:
    """
    Graph the ratio of failed builds.

    Args:
    features_files (list): The features files.
    title (str): The title of the graph.

    Returns:
    None
    """
    fail_percentages = []
    pass_percentages = []
    projects = []

    for file in features_files:
        df = pd.read_csv('prueba/' + file + '.csv')
        total_builds = len(df)
        fail_count = (df['outcome'] == 0).sum()
        pass_count = total_builds - fail_count
        
        # Calcular porcentajes
        fail_percentage = (fail_count / total_builds) * 100
        pass_percentage = (pass_count / total_builds) * 100
        
        fail_percentages.append(fail_percentage)
        pass_percentages.append(pass_percentage)
        
        # Extract project name from file name
        project_name = file.split('/')[-1].split('_')[0]
        projects.append(project_name)

    # Order the data by fail percentage in descending order 
    sorted_indices = sorted(range(len(fail_percentages)), key=lambda i: fail_percentages[i], reverse=True)
    sorted_fail_percentages = [fail_percentages[i] for i in sorted_indices]
    sorted_pass_percentages = [pass_percentages[i] for i in sorted_indices]
    sorted_projects = [projects[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(13, len(sorted_projects) * 0.8))  # Ajustar tamaño del gráfico

    # Graficar cada repositorio
    # Plot each project
    for i, project in enumerate(sorted_projects):
        bar_width = 0.8  # Bar width
        ax.barh(project, sorted_pass_percentages[i], color='green', label='Pass' if i == 0 else "", height=bar_width)
        ax.barh(project, sorted_fail_percentages[i], left=sorted_pass_percentages[i], color='red', label='Fail' if i == 0 else "", height=bar_width)

        # Add target to each bar 
        ax.text(sorted_pass_percentages[i] / 2, project, f"{sorted_pass_percentages[i]:.1f}%", 
                va='center', ha='center', color='black', fontsize=10, fontweight='bold')
        ax.text(sorted_pass_percentages[i] + sorted_fail_percentages[i] / 2, project, f"{sorted_fail_percentages[i]:.1f}%", 
                va='center', ha='center', color='black', fontsize=10, fontweight='bold')

    ax.set_xlabel('Percentage', fontweight='bold')
    ax.set_ylabel('Projects', fontweight='bold') 
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    ax.legend()

    plt.savefig(title + '.svg', format='svg')
    plt.close(fig)

def graph_sensitivities(title1: str, title2: str, sources: np.ndarray) -> None:
    """
    Graph the sensitivities.

    Args:
    sources (np.ndarray): The sources.

    Returns:
    None
    """
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', 'D']
    approaches = ['SBS-Within', 'JAES24-Within', 'JAES24-Without']

    plt.figure(figsize=(10,6))

    # Recall
    for i, source in enumerate(sources):
        file_paths = os.listdir(source)

        dataframes = []

        for file in file_paths:
            df = pd.read_csv(source + file)
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)

        # Group by 'sensitivity' and calculate the mean of 'recall'
        if approaches[i] == 'SBS-Within':
            mean_recall = combined_df.groupby('sensitivity')['recall_rf'].mean().reset_index()
            plt.plot(mean_recall['sensitivity'], mean_recall['recall_rf'], marker = markers[i], color = colors[i], label = approaches[i], markersize=3)
        else:
            mean_recall = combined_df.groupby('sensitivity')['recall_dt'].mean().reset_index()
            plt.plot(mean_recall['sensitivity'], mean_recall['recall_dt'], marker = markers[i], color = colors[i], label = approaches[i], markersize=3)

        plt.xlabel('Sensitivity')
        plt.ylabel('Recall')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.grid(True)
        plt.tight_layout()
        plt.title(title1)
    
    plt.savefig(title1, format='svg')
    plt.close('all')

    # Precision
    for i, source in enumerate(sources):
        file_paths = os.listdir(source)

        dataframes = []

        for file in file_paths:
            df = pd.read_csv(source + file)
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)

        # Group by 'sensitivity' and calculate the mean of 'recall'
        if approaches[i] == 'SBS-Within':
            mean_recall = combined_df.groupby('sensitivity')['precision_rf'].mean().reset_index()
            plt.plot(mean_recall['sensitivity'], mean_recall['precision_rf'], marker = markers[i], color = colors[i], label = approaches[i], markersize=3)
        else:
            mean_recall = combined_df.groupby('sensitivity')['precision_dt'].mean().reset_index()
            plt.plot(mean_recall['sensitivity'], mean_recall['precision_dt'], marker = markers[i], color = colors[i], label = approaches[i], markersize=3)

        plt.xlabel('Sensitivity')
        plt.ylabel('Precision')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.grid(True)
        plt.tight_layout()
        plt.title(title2)
    
    plt.savefig(title2, format='svg')
    plt.close('all')

def graph_recall_precision(title1: str, title2: str, projects_names: list, projects_folders: list, target_techniques: list, source_techniques: list) -> None:
    """
    Graph the recall and precision of several techniques.

    Args:
    title (str): The title of the graph.
    projects_names (list): The names of the projects.
    projects_features_files (list): The features files of the projects.
    target_techniques (list): The target techniques.
    source_techniques (list): The source techniques.

    Returns:
    None
    """
    recalls_sbs = []
    precisions_sbs = []
    recalls_jaes24_within = []
    precisions_jaes24_within = []
    recalls_jaes24_without = []
    precisions_jaes24_without = []
    results_file = '/k-Fold Cross-Validation/Sensitivity/results.csv'

    # Go through each technique
    # SBS-Within, JAES24-Within, JAES24-Without
    for project_folder in projects_folders:
        path_to_csv_sbs = source_techniques[0] + project_folder + results_file
        path_to_csv_jaes24_within = source_techniques[1] + project_folder + results_file
        path_to_csv_jaes24_without = source_techniques[2] + project_folder + results_file
        
        
        df_sbs = pd.read_csv(path_to_csv_sbs)
        df_jaes24_within = pd.read_csv(path_to_csv_jaes24_within)
        df_jaes24_without = pd.read_csv(path_to_csv_jaes24_without)

        recalls_sbs.append(df_sbs['recall_rf'].iloc[0])
        precisions_sbs.append(df_sbs['precision_rf'].iloc[0])

        recalls_jaes24_within.append(df_jaes24_within['recall_dt'].iloc[0])
        precisions_jaes24_within.append(df_jaes24_within['precision_dt'].iloc[0])

        recalls_jaes24_without.append(df_jaes24_without['recall_dt'].iloc[0])
        precisions_jaes24_without.append(df_jaes24_without['precision_dt'].iloc[0])

    # Plot results
    bar_index = np.arange(len(projects_names))
    bar_width = 0.25

    plt.figure(figsize=(10,6))
    plt.bar(bar_index, recalls_sbs, label = target_techniques[0], width=bar_width)
    plt.bar(bar_index + bar_width, recalls_jaes24_within, label = target_techniques[1],width=bar_width)
    plt.bar(bar_index + 2 * bar_width, recalls_jaes24_without, label = target_techniques[2],width=bar_width)

    plt.ylim(0, 1)

    plt.ylabel('Recall (sensitivity=0.5)')
    plt.xticks(bar_index + bar_width, projects_names, rotation=35, ha='center')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title(title1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(title1 + '.svg', format='svg')
    plt.close('all')

    plt.figure(figsize=(10,6))
    plt.bar(bar_index, precisions_sbs, label = target_techniques[0], width=bar_width)
    plt.bar(bar_index + bar_width, precisions_jaes24_within, label = target_techniques[1],width=bar_width)
    plt.bar(bar_index + 2 * bar_width, precisions_jaes24_without, label = target_techniques[2],width=bar_width)

    plt.ylim(0, 1)

    plt.ylabel('Precision (sensitivity=0.5)')
    plt.xticks(bar_index + bar_width, projects_names, rotation=35, ha='center')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title(title2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(title2 + '.svg', format='svg')
    plt.close('all')
