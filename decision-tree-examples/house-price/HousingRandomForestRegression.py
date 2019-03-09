import pandas as pd
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import mean_absolute_error as mea
from sklearn.model_selection import train_test_split as tts


class HousingRandomForestRegression:

    def __init__(self, file):
        self.file = file
        self.data = pd.read_csv(self.file)
        self.fitered_data = self.data.dropna(axis=0)
        self.y = self.fitered_data.Price
        # features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude','Longtitude']
        # self.x = self.fitered_data[features]
        self.x = self.fitered_data.drop(['Price'], axis=1).select_dtypes(exclude=['object'])

    def prediction(self):
        train_x, test_x, train_y, test_y = tts(self.x, self.y, random_state=0)
        model = RF(random_state=1)
        model.fit(train_x, train_y)
        predicted_house_price = model.predict(test_x)
        error = mea(test_y, predicted_house_price)
        print("Mean Absolute Error:  %d" %error)


if __name__ == '__main__':
    my_loader = HousingRandomForestRegression('../data/Melbourne_housing_FULL.csv')
    my_loader.prediction()
