from marshmallow import Schema, fields

class RepositoryMetadataSchema(Schema):
    repository = fields.Str()
    branch = fields.Str()
    features_file = fields.Str()
    pickle_pattern = fields.Str()
    available = fields.Str()
