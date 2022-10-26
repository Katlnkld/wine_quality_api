from flask import Flask, jsonify
from flask_restx import Resource, Api, reqparse, abort
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os


df = pd.read_csv('winequality-red.csv')
df = df[['citric acid','residual sugar','pH','alcohol', 'quality']]
X, y = df.drop('quality', axis=1), df['quality']
#X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=12)

scaler = StandardScaler()

app = Flask(__name__)
api = Api(app,  title='Качество вина', description='Обучение модели и предсказание качества вина')
ns = api.namespace('API', description='Можно задать кислотность, количество сахара, ph и крепость и получить оценку такого вина по шкале от 3 до 8')

fit_parser = reqparse.RequestParser()
fit_parser.add_argument('model_type', type=str, required=True, choices=['classification', 'regression'])

@ns.route('/welcome')
class Wine(Resource):  
    def get(self):
        return "Это моя задача по предсказанию качества вина"


@ns.route("/fitting_wine_model/<string:model_type>")
@ns.doc(params={'model_type': 'Тип используемой модели (classification/regression)'})
class WineModel(Resource):
    #@ns.expect(fit_parser) с этой строкой ничего не работает
    @ns.response(204, 'Модель обучена')
    def post(self, model_type):
        """Обучить модель"""
        X_scaled = scaler.fit_transform(X)
        if model_type=='classification':
            model = LogisticRegression(class_weight='balanced',  max_iter=10000)
        elif model_type=='regression':
            model = LinearRegression()
        else:
            return abort(400, 'Неверный тип модели')

        model.fit(X_scaled, y)
        with open(f'{model_type}_model.pickle', 'wb') as f:
            pickle.dump(model, f)

        with open(f'scaler.pickle', 'wb') as f:
            pickle.dump(scaler, f)
        return None, 204
    
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

        
    
predict_parser = reqparse.RequestParser()
predict_parser.add_argument('model_type', type=str, required=True, choices=['classification', 'regression'])
predict_parser.add_argument('citric_acid', type=float, required=True, help='Доля кислоты (0-1)', default=0.5)
predict_parser.add_argument('residual_sugar', type=float, required=True, help='Остаточный сахар (0-15)')
predict_parser.add_argument('pH', type=float, required=True, help='pH')
predict_parser.add_argument('alcohol', type=float, required=True, help='Крепость')


@ns.route("/prediction/<string:model_type>/<float:citric_acid>/<float:residual_sugar>/<float:pH>/<float:alcohol>")
class prediction(Resource):
    @ns.response(204, 'Предсказания получены')
    @ns.doc(params={'model_type': 'Тип используемой модели (classification/regression)',
    'citric_acid': 'Кислота (0-1)',
    'residual_sugar':'Сахар',
    'pH': 'pH',
    'alcohol':'Крепость'
    })
    #@ns.expect(predict_parser)
    def post(self, model_type,citric_acid,residual_sugar,pH,alcohol):
        """Получить предсказания"""

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