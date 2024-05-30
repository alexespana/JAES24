import os
import pandas as pd
from flask import Flask, request
from concurrent.futures import ThreadPoolExecutor
from models import db, RepositoryMetadata
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import get_owner, get_repo_name, is_repository_url
from github_manager import GithubManager
from features_manager import ( 
    get_performance_short, get_performance_long, 
    get_time_frequency, get_num_commits, 
    get_num_files_changed, get_num_lines_changed, 
    get_failure_distance, get_weekday, get_hour,
    get_outcome, get_features
)

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=4)

app.config["SQLALCHEMY_DATABASE_URI"] = 'postgresql://' + os.getenv('POSTGRES_USER') + ':' + os.getenv('POSTGRES_PASSWORD') + '@db:5432/' + os.getenv('POSTGRES_DB')
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()

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
        repository_url = request.form['repository_url']
        branch = request.form['branch']

        if is_repository_url(repository_url):
            owner = get_owner(repository_url)
            repo_name = get_repo_name(repository_url)

            # Get CI builds to train the model
            github_manager = GithubManager()
            builds = github_manager.get_builds(owner=owner, repo_name=repo_name, branch=branch, number_of_builds=10)

            df = pd.DataFrame(columns=['PS', 'PL', 'TF', 'NC', 'FC', 'FA', 'FM', 'FR', 'LC', 'LA', 'LR', 'LT', 'UT' ,'FD', 'WD', 'DH', 'outcome'])

            # Get the data to train the model
            for build in builds["workflow_runs"]:
                build_id = build['id']
                app.logger.info(str(build_id))

                PS = get_performance_short(build_id, builds)
                PL = get_performance_long(build_id, builds)
                TF = get_time_frequency(build_id, builds)
                NC = get_num_commits(build)
                FC, FA, FM, FR = get_num_files_changed(build)
                LC, LA, LR, LT, UT = get_num_lines_changed(build)
                FD = get_failure_distance(build_id, builds)
                WD = get_weekday(build)
                DH = get_hour(build)
                outcome = get_outcome(build)

                # Add CI build
                df.loc[len(df.index)] = [PS, PL, TF, NC, FC, FA, FM, FR, LC, LA, LR, LT, UT, FD, WD, DH, outcome]

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

            # Train the model - Decision Tree Classifier
            model = DecisionTreeClassifier(random_state=42)
            model.fit(x_train, y_train)

            # Make some predictions
            predictions = model.predict(x_test)

            app.logger.info('Predictions for: ' + repository_url)

            # Add a column outcome with the predictions
            x_test['outcome'] = predictions

            # Show the predictions
            app.logger.info("\n" + str(x_test))

            # Calculate the accuracy comparing the predictions with the t_test
            accurary = predictions == y_test
            app.logger.info('Accuracy: ' + str(accurary.mean()))

            return 'Predictions made'
        else:
            return 'Invalid GitHub URL'

@app.route('/repository_metadata/create', methods=['POST'])
def save_builds():

    if request.method == 'POST':
        repository_url = request.form['repository_url']

        if is_repository_url(repository_url):
            repo_name = get_repo_name(repository_url)
            branch = request.form['branch'].replace('/', '-')
            features_file = repo_name + '_' + branch + '.csv'

            repo_metadata = RepositoryMetadata(
                repository = repo_name,
                branch = branch,
                features_file = features_file
            )

            db.session.add(repo_metadata)
            db.session.commit()

            # Async call to fetch information from the specified repository and branch            
            executor.submit(get_features,  repository_url, branch, features_file)

            return 'Repository metadata created'
        else:
            return 'Invalid GitHub URL'
