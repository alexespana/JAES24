"""
This file contains all the constants used in the project.
"""
FEATURES_FOLDER = 'features/'
AIMODELS_FOLDER = 'AIModels/'

# API urls   
# Lists pull requests in a specified repository
GET_PRS = 'https://api.github.com/repos/OWNER/REPO/pulls'
# Lists details of a pull request by providing its number
GET_PR = 'https://api.github.com/repos/OWNER/REPO/pulls/PULL_NUMBER'
# Lists a maximum of 250 commits for a pull request
GET_PR_COMMITS = 'https://api.github.com/repos/OWNER/REPO/pulls/PULL_NUMBER/commits'
# Lists the files in a specified pull request
GET_PR_FILES = 'https://api.github.com/repos/OWNER/REPO/pulls/PULL_NUMBER/files'
# Lists all workflow runs for a repository
GET_BUILDS = 'https://api.github.com/repos/OWNER/REPO/actions/runs'
# Gets a specific workflow run
GET_BUILD = 'https://api.github.com/repos/OWNER/REPO/actions/runs/RUN_ID'
# Get a specific commit
GET_COMMIT = 'https://api.github.com/repos/OWNER/REPO/commits/COMMIT_SHA'
# Converter to obtain the time frequency in hours
HOUR_CONVERTER = 3600
# Time to retry the request in seconds
RETRY_TIME = 60

# Model classifiers types
DT_CLASSIFIER = 'Decision Tree'
RF_CLASSIFIER = 'Random Forest'
LR_CLASSIFIER = 'Logistic Regression'
SVM_CLASSIFIER = 'Support Vector Machine'
KNN_CLASSIFIER = 'K-Nearest Neighbors'
NN_CLASSIFIER = 'Neural Network'

