import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template
from flask_restx import Api, Resource, abort, reqparse
from flask_sqlalchemy import SQLAlchemy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

data = np.array([0.1, 0.2, 0.3, 0.4]).reshape(1, -1)
print(jsonify(data))