import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

##https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

class TemperaturePredictionRF:

    def __init__(self):
        features = pd.read_csv('../../data/weather.csv')
        features = pd.get_dummies(features)
        self.features_list = list(features.columns)
        self.labels = np.array(features['actual'])
        features = features.drop('actual', axis=1)
        self.features = np.array(features)

    def train(self):
        train_ft, test_ft, train_l, test_l = train_test_split(self.features,self.labels, test_size=0.25, random_state=42)
        rf = RandomForestRegressor(n_estimators=150,random_state=42)
        rf.fit(train_ft, train_l)
        predictions = rf.predict(test_ft)
        print('Mean abs error : ', mean_absolute_error(test_l, predictions) , ' degrees')


if __name__ == '__main__':
    regressor = TemperaturePredictionRF()
    regressor.train()
