import pandas as pd
from sklearn.metrics import mean_absolute_error as mea
from sklearn.model_selection import train_test_split as tts
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor as GB


class HousingXGBoostRegression:

    def __init__(self, file):
        self.file = file
        self.data = pd.read_csv(self.file)
        self.fitered_data = self.data.dropna(axis=0)
        self.y = self.fitered_data.Price
        # features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude','Longtitude']
        # self.x = self.fitered_data[features]
        self.x = self.fitered_data.drop(['Price'], axis=1).select_dtypes(exclude=['object'])

    def prediction(self):
        train_x, test_x, train_y, test_y = tts(self.x.as_matrix(), self.y.as_matrix(), random_state=0)
        my_imputer = SimpleImputer()
        train_x = my_imputer.fit_transform(train_x)
        test_x = my_imputer.transform(test_x)
        my_model = GB(n_estimators=1000, learning_rate=0.20)
        my_model.fit(train_x, train_y, early_stopping_rounds=15, eval_set=[(test_x, test_y)], verbose=False)
        error = mea(test_y, my_model.predict(test_x))
        print("Mean Absolute Error:  %d" %error)


if __name__ == '__main__':
    my_loader = HousingXGBoostRegression('../data/Melbourne_housing_FULL.csv')
    my_loader.prediction()
