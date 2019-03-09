import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


class GasCosptRF:

    def __init__(self, file):
        self.dataset = pd.read_csv(file)
        self.y = self.dataset.Petrol_Consumption
        self.x = self.dataset.drop(['Petrol_Consumption'], axis=1).select_dtypes(exclude=['object'])
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y,test_size=0.2, random_state=0)
        sc = StandardScaler()
        self.train_x = sc.fit_transform(self.train_x)
        self.test_x = sc.transform(self.test_x)

    def train(self, n_estimators):

        regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
        regressor.fit(self.train_x, self.train_y)
        predicted_gaz_consumption = regressor.predict(self.test_x)
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.test_y, predicted_gaz_consumption))
        # print('Mean Squared Error:', metrics.mean_squared_error(test_y, predicted_gaz_consumption))
        # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, predicted_gaz_consumption)))


if __name__ == '__main__':
    my_service = GasCosptRF('../../data/petrol_consumption.csv')
    for n_estimators in [5, 10, 50, 100, 150, 200, 250]:
        print(n_estimators)
        my_service.train(n_estimators)
