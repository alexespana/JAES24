"""
This file contains all the constants used in the project.
"""
STATIC_FOLDER = 'static/'
REPOSITORIES_CSV_FILE = 'repositories.csv'

CI_JUNIT = 'ci-junit.csv'
REPOSITORY_URL = 'https://github.com/junit-team/junit4'

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
