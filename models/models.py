from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

import json

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, Ridge, Perceptron, Lasso
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score

model_types = {'classification' : 'LogisticRegression, RidgeClassifier, Perceptron',
               'regression' : 'LinearRegression, Ridge, Lasso'}

possible_models = {
    'LinearRegression' : (LinearRegression(), mean_absolute_error, 'MAE'),
    'LogisticRegression' : (LogisticRegression(class_weight='balanced',  max_iter=10000), balanced_accuracy_score, 'BalancedAcc'),
    'RidgeClassifier' : (RidgeClassifier(class_weight='balanced'), balanced_accuracy_score, 'BalancedAcc'),
    'Ridge' : (Ridge(), mean_absolute_error, 'MAE'),
    'Lasso' : (Lasso(), mean_absolute_error, 'MAE'),
    'Perceptron' : (Perceptron(class_weight='balanced'), balanced_accuracy_score, 'BalancedAcc')
}

class WineQualityModel:
    def __init__(self, wine_type, model_type, hyper_params):
  
        self.wine_type = wine_type    
        self.model_type= model_type
        self.hyper_params = hyper_params
        self.scaler = StandardScaler()

    def initialize(self):
    
        df = pd.read_csv(f'data/winequality-{self.wine_type}.csv', sep=';')
        
        data, target = df.drop('quality', axis=1), df['quality']
       
        self.train_data,  self.val_data, self.train_target, self.val_target = \
                    train_test_split(data, target, stratify=target, random_state=12)

        self.train_data[self.train_data.columns] = self.scaler.fit_transform(self.train_data)
        self.val_data[self.val_data.columns]  = self.scaler.transform(self.val_data)
    
    def fit(self):
        self.model = possible_models[self.model_type][0]
        if self.hyper_params is not None:
            self.model.set_params(**self.hyper_params)
        self.model.fit(self.train_data, self.train_target)

        return self.model

    def evaluate(self):
        metric = possible_models[self.model_type][1]
        quality = metric(self.val_target, self.model.predict(self.val_data))
        return possible_models[self.model_type][2], quality

    def train_pipeline(self):
        self.initialize()
        model = self.fit()
        scaler = self.scaler
        return model, scaler


def predict(data, model, scaler):
    data = json.loads(data)
    data = pd.read_json(json.dumps(data))
    data[data.columns] = scaler.transform(data)
    my_prediction = model.predict(data)
    return my_prediction



#wine_quality_model = WineQualityModel(wine_type = 'red', model_type = 'Perceptron', hyper_params=None)
#model, scaler = wine_quality_model.train_pipeline()
#print(model)