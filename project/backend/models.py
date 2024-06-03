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

