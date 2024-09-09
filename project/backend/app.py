import os
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from models import db, RepositoryMetadata, init_db
from sklearn.model_selection import train_test_split
from flask_marshmallow import Marshmallow
from utils import (
    get_repo_name, is_repository_url,
    normalize_branch_name, is_csv_available,
    get_builds_folder, get_aimodels_folder
)
from model_manager import is_model_available, get_model_path
from features_manager import  process_repository
from flask_cors import CORS
from schemas import RepositoryMetadataSchema
from constants import NN_CLASSIFIER

app = Flask(__name__)
ma = Marshmallow(app)
CORS(app)       # Do it more restrictive in production

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
            _, x_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if is_model_available(repository_metadata.pickle_pattern, classifier):
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)

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
                return 'The model has not been trained yet, please be patient.'
        else:
            return 'We are still gathering information from your repository. Please check back in a few minutes.'

@app.route('/api/repository', methods=['POST'])
def save_builds():

    if request.method == 'POST':
        repository_url = request.form['repository_url']

        if is_repository_url(repository_url):
            repo_name = get_repo_name(repository_url)
            branch = request.form['branch']
            features_file = repo_name + '_' + normalize_branch_name(branch) + '.csv'
            pickle_pattern = repo_name + '_' + normalize_branch_name(branch)

            # Search the repository in the database
            repository_metadata = RepositoryMetadata.query.filter_by(repository=repo_name, branch=branch).first()
            if repository_metadata:
                return jsonify({'message': 'The repository and branch you have entered already exist.'}), 200
            else:
                repo_metadata = RepositoryMetadata(
                    repository = repo_name,
                    branch = branch,
                    features_file = features_file,
                    pickle_pattern = pickle_pattern
                )

                db.session.add(repo_metadata)
                db.session.commit()

                # Create a folder with the repository name and branch
                os.makedirs(get_builds_folder(repo_name, branch), exist_ok=True)
                os.makedirs(get_aimodels_folder(repo_name, branch), exist_ok=True)

                # Async call to fetch information from the specified repository and branch
                executor.submit(process_repository, repository_url, branch, features_file, pickle_pattern)   

            return jsonify({'message': 'The repository and branch have been saved successfully.'}), 201
        else:
            return jsonify({'error': 'Invalid repository URL'}), 400

@app.route('/api/available-models', methods=['GET'])
def available_models():
    models = RepositoryMetadata.query.all()
    schema = RepositoryMetadataSchema(many=True)

    # Go throuh them and check if the last model which is created is available
    for model in models:
        available = is_model_available(model.pickle_pattern, NN_CLASSIFIER)
        model.available = 'Yes' if available else 'No'

    models = RepositoryMetadata.query.all()
    result = schema.dump(models)
    if models:
        return jsonify(result), 200
    else:
        return jsonify({'message': 'No models available.'}), 200
