from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Подключение к базе данных
class ModelDatabase(db.Model):
    __tablename__ = "db_models"

    id = db.Column(db.Integer, primary_key=True)
    wine_type = db.Column(db.String, nullable=False)
    model_type = db.Column(db.String, nullable=False)
    hyper_params = db.Column(db.String, nullable=False)
    quality = db.Column(db.Float, nullable=False)
    model_dump = db.Column(db.LargeBinary, nullable=False)
    scaler_dump = db.Column(db.LargeBinary, nullable=False)
    
def check_existence(ModelDatabase, wine_type, model_type, hyper_params):
    exists = db.session.query(db.exists().where(
            (ModelDatabase.wine_type  == wine_type) &
            (ModelDatabase.model_type == model_type) &
            (ModelDatabase.hyper_params == hyper_params)
            )).scalar()
    return exists

def get_value_from_db(column, wine_type, model_type, hyper_params):
    value = db.session.query(column).where(
                (ModelDatabase.wine_type  == wine_type) &
                (ModelDatabase.model_type == model_type) &
                (ModelDatabase.hyper_params == hyper_params)
                ).one()
    return value