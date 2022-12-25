import pickle
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, request
from flask_restx import Api, Resource, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm.exc import NoResultFound
from utils.database import ModelDatabase, check_existence, get_value_from_db
from models.models import WineQualityModel, predict, model_types

app = Flask(__name__)
db = SQLAlchemy()

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

api = Api(app,  title='Качество вина', description='Обучение модели и предсказание качества вина')
ns = api.namespace('API', description='')
db = SQLAlchemy(app)

def create_app(config_object):
    app = Flask(__name__)
    app.config.from_object(config_object)
    db.init_app(app)
    db = SQLAlchemy(app)


    

@ns.route("/possiblemodels")
@ns.doc(params={ 'task_type' : 'Тип модели (classification/regression)'})
class PossibleModels(Resource):
    def get(self):
        try:
            task_type = request.args.get('task_type')
            return jsonify({ f'Possible models for {task_type}': model_types[task_type]})
        except KeyError:
            return abort(400, 'Not a type')


@ns.route("/getorcreate")
@ns.doc(params={ 'wine_type' : 'Тип вина (красное/белое)',
    'model_type': 'Тип используемой модели (из списка с моделями)',
    'hyper_params' : 'Гиперпараметры модели'})
class CreateOrGetModel(Resource):
    @ns.response(204, 'Модель обучена')
    def get(self):
        """Обучить модель"""
        self.wine_type = request.args.get('wine_type')
        self.model_type = request.args.get('model_type')
        self.hyper_params = request.args.get('hyper_params')

        exists = check_existence(ModelDatabase, wine_type = self.wine_type, model_type = self.model_type, \
                            hyper_params=self.hyper_params)
        if not exists:
            # создаем и загружаем модель
            wine_quality_model = WineQualityModel(wine_type = self.wine_type, model_type = self.model_type, \
                            hyper_params=self.hyper_params)
            model, scaler = wine_quality_model.train_pipeline()
            encoded_model =  pickle.dumps(model)
            encoded_scaler = pickle.dumps(scaler)
            
            metric, quality = wine_quality_model.evaluate()

            query = ModelDatabase(
                wine_type = self.wine_type,
                model_type = self.model_type,
                hyper_params = self.hyper_params,
                model_dump = encoded_model,
                scaler_dump = encoded_scaler,
                quality = quality
                )
            db.session.add(query)
            db.session.commit()
            return jsonify(status='succsess', message='Модель успешно обучена',
            metric = metric, quality = quality)
        else:
            return abort(400, 'Такая модель уже существует')

    @ns.response(204, 'Успешно удалено')
    def delete(self):
        """Удалить модель"""
        self.wine_type = request.args.get('wine_type')
        self.model_type = request.args.get('model_type')
        self.hyper_params = request.args.get('hyper_params')

        try:
            obj = db.session.query(ModelDatabase).where(
                (ModelDatabase.wine_type  == self.wine_type) &
                (ModelDatabase.model_type == self.model_type) &
                (ModelDatabase.hyper_params == self.hyper_params)
                ).one()
            db.session.delete(obj)
            db.session.commit()
            return jsonify(status='succsess', message='Модель удалена')
        except NoResultFound:
            return abort(400, 'Модель еще не обучена')

        
@ns.route("/getprediction")
class GetPrediction(Resource):
    @ns.response(204, 'Предсказания получены')
    @ns.doc(params={  'wine_type' : 'Тип вина (красное/белое)',
    'model_type': 'Тип используемой модели (из списка с моделями)',
    'hyper_params' : 'Гиперпараметры модели' ,
    'data' : 'Данные для получения предсказаний'})
    def post(self):
        """Получить предсказания"""
        self.wine_type = request.args.get('wine_type')
        self.model_type = request.args.get('model_type')
        self.hyper_params = request.args.get('hyper_params')
        self.data = request.args.get('data')

        exists = check_existence(ModelDatabase, wine_type = self.wine_type, model_type = self.model_type, \
                            hyper_params=self.hyper_params)

        if exists:
            encoded_model = get_value_from_db(ModelDatabase.model_dump, self.wine_type, 
                            self.model_type, self.hyper_params)
            encoded_scaler = get_value_from_db(ModelDatabase.scaler_dump, self.wine_type, 
                            self.model_type, self.hyper_params)

            model= pickle.loads(encoded_model[0])
            scaler = pickle.loads(encoded_scaler[0])

            my_prediction = predict(self.data, model, scaler)

            return jsonify(prediction = np.round(my_prediction).tolist())
        else: 
            return abort(400, 'Модель еще не обучена')


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5432, debug=True)




