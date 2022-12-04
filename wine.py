import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, make_response
from flask_restx import Api, Resource, abort, reqparse
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error, balanced_accuracy_score

scaler = StandardScaler()

app = Flask(__name__)
db = SQLAlchemy()

app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://localhost/Wine"
db.init_app(app)

api = Api(app,  title='Качество вина', description='Обучение модели и предсказание качества вина')
ns = api.namespace('API', description='Можно задать кислотность, количество сахара, ph и крепость и получить оценку такого вина по шкале от 3 до 8')


db = SQLAlchemy(app)

# Подключение к базе данных
class WineParams(db.Model):
    __tablename__ = "requests_data"

    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String, nullable=False)
    citric_acid = db.Column(db.Float, nullable=False)
    residual_sugar = db.Column(db.Float, nullable=False)
    pH = db.Column(db.Float, nullable=False)
    alcohol = db.Column(db.Float, nullable=False)    


@ns.route('/welcome')
class Wine(Resource):  
    def get(self):
        print("Это моя задача по предсказанию качества вина")
        return make_response(render_template('index.html'))

#fit_parser = reqparse.RequestParser()
#fit_parser.add_argument('model_type', type=str, required=True, choices=['classification', 'regression'])

@ns.route("/fit")
@ns.doc(params={ 'wine_type' : 'Тип вина (красное/белое)',
    'model_type': 'Тип используемой модели (classification/regression)'})
class Fit(Resource):
    #@ns.expect(fit_parser) с этой строкой ничего не работает
    @ns.response(204, 'Модель обучена')
    def get(self):
        """Обучить модель"""
        self.wine_type = request.form.get('wine_type')
        self.model_type = request.form.get('model_type')

        if str(self.wine_type).lower() in (['белое', 'white']):
            df = pd.read_csv('winequality-white.csv', sep=';')
        elif str(self.wine_type).lower() in (['красное', 'red']):
            df = pd.read_csv('winequality-red.csv')
        else:
            abort(400, 'Такого вина у нас нет. Попробуйте красное или белое.')

        df = df[['citric acid','residual sugar','pH','alcohol', 'quality']]
        X, y = df.drop('quality', axis=1), df['quality']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, random_state=12)

        X_scaled = scaler.fit_transform(self.X_train)
        if self.model_type=='classification':
            model = LogisticRegression(class_weight='balanced',  max_iter=10000)
        elif self.model_type=='regression':
            model = LinearRegression()
        else:
            return abort(400, 'Неверный тип модели')

        model.fit(X_scaled, self.y_train)

        """Сохранить модель"""
        with open(f'{self.model_type}_model.pickle', 'wb') as f:
            pickle.dump(model, f)

        with open(f'scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)

        """Проверить модель"""
        if self.model_type=='classification':
            quality = balanced_accuracy_score(self.y_test, model.predict(self.X_test))
            return jsonify(accuracy = quality)

        elif self.model_type=='regression':
            quality = mean_absolute_error(self.y_test, model.predict(self.X_test))
            return jsonify(mae = quality)
        else:
            return abort(400, 'Неверный тип модели')

    @ns.response(204, 'Модель успешно удалена')
    def delete(self, model_type):
        """Удалить обученную модель"""
        if model_type not in ['classification','regression']:
            return abort(400, 'Неверный тип модели')
        try:
            os.remove(f'{model_type}_model.pickle')
            return None, 204
        except FileNotFoundError:
            return abort(400, 'Модель еще не обучена')

        
    
#predict_parser = reqparse.RequestParser()
#predict_parser.add_argument('model_type', type=str, required=True, choices=['classification', 'regression'])
#predict_parser.add_argument('citric_acid', type=float, required=True, help='Доля кислоты (0-1)', default=0.5)
#predict_parser.add_argument('residual_sugar', type=float, required=True, help='Остаточный сахар (0-15)')
#predict_parser.add_argument('pH', type=float, required=True, help='pH')
#predict_parser.add_argument('alcohol', type=float, required=True, help='Крепость')


@ns.route("/prediction")  #/<string:model_type>/<float:citric_acid>/<float:residual_sugar>/<float:pH>/<float:alcohol>")
class prediction(Resource):
    @ns.response(204, 'Предсказания получены')
    @ns.doc(params={'model_type': 'Тип используемой модели (classification/regression)',
    'citric_acid': 'Кислота (0-1)',
    'residual_sugar':'Сахар',
    'pH': 'pH',
    'alcohol':'Крепость'
    })
    #@ns.expect(predict_parser)
    def post(self): #, model_type,citric_acid,residual_sugar,pH,alcohol):
        """Получить предсказания"""
        model_type = request.args.get('model_type')
        citric_acid = request.args.get('citric_acid')
        residual_sugar = request.args.get('residual_sugar')
        pH = request.args.get('pH')
        alcohol = request.args.get('alcohol')

        #query = WineParams(
            #model_type=request.form["model_type"],
        #    citric_acid=request.form["citric_acid"],
        #    residual_sugar=request.form["residual_sugar"],
        #    pH=request.form["pH"],
        #    alcohol=request.form["alcohol"],
        #)
        query = WineParams(model_type = model_type, citric_acid = citric_acid, residual_sugar = residual_sugar, 
        pH = pH, alcohol = alcohol)
        db.session.add(query)
        db.session.commit()
        print('Запрос добавлен')

        if model_type not in ['classification','regression']:
            return abort(404, 'Неверный тип модели')

        model = pickle.load(open(f'{model_type}_model.pickle', 'rb'))
        scaler = pickle.load(open(f'scaler.pickle', 'rb'))
        data = np.array([citric_acid,residual_sugar,pH,alcohol]).reshape(1, -1)
        data = scaler.transform(data)
        my_prediction = model.predict(data)
        return jsonify(prediction = round(float(my_prediction)))


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5432, debug=True)