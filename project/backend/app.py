from flask import Flask, request
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Define a constant with the url repository
STATIC_FOLDER = 'static/'
REPOSITORIES_CSV_FILE = 'repositories.csv'

CI_JUNIT = 'ci-junit.csv'
REPOSITORY_URL = 'https://github.com/junit-team/junit4'

#######################################
# AUXILIARY FUNCTIONS
#######################################
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

def is_a_known_repository(url):
    """
    Check if the given repository URL is present in the repositories CSV file.

    Args:
    repository_url (str): The URL of the repository to check.

    Returns:
    bool: True if the repository is present in the CSV file, False otherwise.
    """
    result = False
    repositories = pd.read_csv(STATIC_FOLDER + REPOSITORIES_CSV_FILE)

    # Remove from the url the https://github.com/
    name = url.replace('https://github.com/', '')

    if name in repositories['name'].values:
        result = True

    return result

#######################################

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
        repository_url = request.form['repository_url']

        # Check if the URL is a GitHub repository URL
        if not is_repository_url(repository_url):
            return 'Invalid GitHub URL'

        # Check if it is present in the repocisoties csv
        if is_a_known_repository(repository_url):
            data = pd.read_csv(STATIC_FOLDER + CI_JUNIT)

            # Features and target
            X = data[['num_lines', 'num_files', 'num_commits']]
            y = data['outcome']

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            # Train the model - Decision Tree Classifier
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Make some predictions 
            predictions = model.predict(X_test)

            app .logger.info('Predictions for: ' + REPOSITORY_URL)

            # Add a column outcome with the predictions
            X_test['outcome'] = predictions

            # Show the predictions
            app.logger.info("\n" + str(X_test))

            # Calculate the accuracy comparing the predictions with the t_test
            accurary = predictions == y_test
            app.logger.info('Accuracy: ' + str(accurary.mean()))

            return('Repository is known')
        else:       # Use the gigawork command line tool to get repository information
            return('Repository is unknown')
