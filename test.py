from api import app
import unittest
from unittest import TestCase
from flask_sqlalchemy import SQLAlchemy
import mock
from unittest.mock import patch
from flask import Flask
from utils.database import ModelDatabase, check_existence, get_value_from_db
import json
from unittest.mock import Mock
from utils.utils import get_sample_json
from werkzeug.exceptions import NotFound


class TestPossibleModels(TestCase):
    def setUp(self):
        app.config['TESTING'] = True 
        self.app = app.test_client()

    def test_classification(self):
        response = self.app.get('API/possiblemodels', query_string=dict(task_type='classification'))
        response_json = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_json['Possible models for classification'], 'LogisticRegression, RidgeClassifier, Perceptron')

    def test_regression(self):
        response = self.app.get('API/possiblemodels', query_string=dict(task_type='regression'))
        response_json = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_json['Possible models for regression'], 'LinearRegression, Ridge, Lasso')

    def test_bad_request(self):
        response = self.app.get('API/possiblemodels', query_string=dict(task_type='something_else'))
        response_json = json.loads(response.data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response_json['message'], 'Not a type')



@mock.patch("flask_sqlalchemy.SignallingSession", autospec=True)
class TestCreateModel(TestCase):
    def create_app(self):
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        app.config['TESTING'] = True 
        self.db = SQLAlchemy()
        self.db.init_app(app)

        with app.app_context():
            self.db.create_all()
            
        return app

    def setUp(self):
        self.app = self.create_app().test_client()

    def test_inserting_to_db(self, session):
 
        with app.app_context():
            self.query = ModelDatabase(
                wine_type = 'white',
                model_type = 'Ridge',
                )
            self.db.session.add(self.query)
            self.db.session.commit()
       
            self.assertEqual(check_existence(ModelDatabase, 'white', 'Ridge', None), True)
        
    def tearDown(self):
        
        self.db.session.query(ModelDatabase).filter((ModelDatabase.wine_type=='white') & 
                                (ModelDatabase.model_type == 'Ridge')).delete()
        self.db.session.commit()
       
class TestGetPrediction(TestCase):
    def setUp(self):
        app.config['TESTING'] = True 
        self.app = app.test_client()
        self.data = get_sample_json('red', 5)
        #print(self.data)

    def test_prediction(self):
        response = self.app.post(f'API/getprediction?wine_type=red&model_type=Perceptron&data={self.data}')
        response_json = json.loads(response.data)
        #print(response_json['prediction'])

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response_json['prediction'], list)
        self.assertEqual(len(response_json['prediction']), 5)
       


if __name__ == "__main__":
    unittest.main()



 