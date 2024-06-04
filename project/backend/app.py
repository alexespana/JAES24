import os
import pandas as pd
import pickle
from flask import Flask, request
from concurrent.futures import ThreadPoolExecutor
from models import db, RepositoryMetadata, init_db
from sklearn.model_selection import train_test_split
from utils import (
    get_repo_name, is_repository_url,
    normalize_branch_name, is_csv_available,
    is_model_available, get_model_path,
    get_model
)
from utils import get_repo_name, is_repository_url, normalize_branch_name, is_csv_available, is_model_available, get_model_path, get_model
from features_manager import  get_features

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

app.config["SQLALCHEMY_DATABASE_URI"] = 'postgresql://' + os.getenv('POSTGRES_USER') + ':' + os.getenv('POSTGRES_PASSWORD') + '@db:5432/' + os.getenv('POSTGRES_DB')
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    init_db()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the outcome of a build based on the features.
    Features:
        PS: performance short
        PL: performance long
        TF: time frequency
        NC: number of commits
        FC: number of files changed
        FA: number of files added
        FM: number of files modified
        FR: number of files removed
        LC: number of lines changed
        LA: number of lines added
        LR: number of lines removed
        LT: number of tests lines changed
        UT: a number indicating whether tests have been written
        FD: failure distance
        WD: week day
        DH: day hour
    Class:
        outcome: success or failure
    """
    if request.method == 'POST':
            
        data = request.json
        repo_name = data['repository']
        branch = data['branch']
        classifier = data['classifier']
        
        repository_metadata = RepositoryMetadata.query.filter_by(repository=repo_name, branch=branch).first()
        if is_csv_available(repository_metadata.features_file):
            df = pd.read_csv('features/' + repository_metadata.features_file)
            model_path = get_model_path(repository_metadata.pickle_pattern, classifier)

            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)

            app.logger.info(df)
            # Features and target
            X = df[['PS', 'PL', 'TF', 'NC', 'FC', 'FA', 'FM', 'FR', 'LC', 'LA', 'LR', 'LT', 'UT', 'FD', 'WD', 'DH']]
            y = df['outcome']

            # Split the data into training and test sets
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            if is_model_available(repository_metadata.pickle_pattern, classifier):
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
            else:
                model = get_model(classifier)

                # Train
                model.fit(x_train, y_train)

                # Save the model (pickle file)
                with open(model_path, 'wb') as file:
                    pickle.dump(model, file)

            # Make some predictions
            predictions = model.predict(x_test)

            app.logger.info('Predictions for: ' + repo_name + ' - ' + branch)

            # Add a column outcome with the predictions
            x_test['outcome'] = predictions

            # Show the predictions
            app.logger.info("\n" + str(x_test))

            # Calculate the accuracy comparing the predictions with the t_test
            accurary = predictions == y_test
            app.logger.info('Accuracy: ' + str(accurary.mean()))

            return "Predictions for: " + repo_name + " - " + branch + "\n" + str(x_test)
        else:
            return 'We are still gathering information from your repository. Please check back in a few minutes.'

@app.route('/repository_metadata/create', methods=['POST'])
def save_builds():

    if request.method == 'POST':
        repository_url = request.form['repository_url']

        if is_repository_url(repository_url):
            repo_name = get_repo_name(repository_url)
            branch = request.form['branch']
            features_file = repo_name + '_' + normalize_branch_name(branch) + '.csv'
            pickle_pattern = repo_name + '_' + normalize_branch_name(branch)

            repo_metadata = RepositoryMetadata(
                repository = repo_name,
                branch = branch,
                features_file = features_file,
                pickle_pattern = pickle_pattern
            )

            db.session.add(repo_metadata)
            db.session.commit()

            # Async call to fetch information from the specified repository and branch            
            executor.submit(get_features,  repository_url, branch, features_file)

            return 'Repository metadata created'
        else:
            return 'Invalid GitHub URL'
