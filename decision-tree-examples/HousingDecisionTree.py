import pandas as pd
from sklearn.tree import DecisionTreeRegressor as DT
from sklearn.metrics import mean_absolute_error as mea
from sklearn.model_selection import train_test_split as tts


class HousingDecisionTree:

    def __init__(self, file):
        self.file = file
        self.data = pd.read_csv(self.file)
        self.fitered_data = self.data.dropna(axis=0)
        #Target prediction
        self.y = self.fitered_data.Price
        #Features
        features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude','Longtitude']
        self.x = self.fitered_data[features]

    def defineMaxLeafNodes(self, max_leaf_nodes, train_x, test_x, train_y, test_y):
        model = DT(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_x, train_y)
        predicted_house_price = model.predict(test_x)
        error = mea(test_y, predicted_house_price)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, error))

    def prediction(self):
        train_x, test_x, train_y, test_y = tts(self.x, self.y, random_state=0)
        # Determine Low Point between Underfitting/Overfitting
        for max_leaf_nodes in [2, 5, 50, 100, 500, 1000, 5000]:
            self.defineMaxLeafNodes(max_leaf_nodes, train_x, test_x, train_y, test_y)


if __name__ == '__main__':
    my_loader = HousingDecisionTree('../data/Melbourne_housing_FULL.csv')
    my_loader.prediction()
