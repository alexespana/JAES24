from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
  pass

db = SQLAlchemy(model_class=Base)

class RepositoryMetadata(db.Model):
    __tablename__ = 'repository_metadata'
    __table_args__ = (
        db.PrimaryKeyConstraint('repository', 'branch'),
    )

    repository = db.Column(db.String(255), primary_key=True)
    branch = db.Column(db.String(255), primary_key=True)
    features_file = db.Column(db.String(255))
    pickle_pattern = db.Column(db.String(255))

class LookupModels(db.Model):
   __tablename__ = 'lookup_models'

   name = db.Column(db.String(255), primary_key=True)
   description = db.Column(db.String(255))


def init_db():
    db.create_all()  # Creates all tables defined by models
    if not LookupModels.query.first():
        # Insert the AI models
        dt = LookupModels(name='Decision Tree', description='Supervised learning models that split data into branches to make decisions based on feature values.')
        db.session.add(dt)
        rd = LookupModels(name='Random Forest', description='Ensemble method combining multiple decision trees to improve accuracy.')
        db.session.add(rd)
        lg = LookupModels(name='Linear Regression', description='Models the relationship between dependent and independent variables using a linear equation.')
        db.session.add(lg)
        svm = LookupModels(name='Support Vector Machine', description='Classifies data by finding the optimal hyperplane that separates classes.')
        db.session.add(svm)
        kn = LookupModels(name='K-Nearest Neighbors', description='Classifies data based on the "k" closest training examples.')
        db.session.add(kn)
        nn = LookupModels(name='Neural Network', description='Models complex patterns using layers of interconnected nodes.')
        db.session.add(nn)

        # Save on DB
        db.session.commit()
