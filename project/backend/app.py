from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import get_owner, get_repo_name, is_repository_url
from github_manager import GithubManager

app = Flask(__name__)

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

@app.route('/get_prs', methods=['GET'])
def get_pull_requests():
    if request.method == 'GET':
        repository_url = request.form['repository_url']

        if is_repository_url(repository_url):
            owner = get_owner(repository_url)
            repo_name = get_repo_name(repository_url)

            # Make the request
            github_manager = GithubManager()
            pull_requests = github_manager.get_pull_requests(owner, repo_name)

            return pull_requests
        else:
            return 'Invalid GitHub URL'
