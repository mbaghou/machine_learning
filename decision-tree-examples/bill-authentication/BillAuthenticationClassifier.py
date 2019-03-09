import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class BillAuthenticationClassifier:

    def __init__(self, file):
        data = pd.read_csv(file)
        self.x = data.drop(['Class'], axis=1).select_dtypes(exclude=['object'])
        self.y = data.Class
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.x, self.y, test_size=0.2,random_state=0)
        sc = StandardScaler()
        self.train_x = sc.fit_transform(self.train_x)
        self.test_x = sc.transform(self.test_x)

    def train(self):
        self.classifier = RandomForestClassifier(n_estimators=20, random_state=0)
        self.classifier.fit(self.train_x, self.train_y)
        class_predict = self.classifier.predict(self.test_x)
        print(confusion_matrix(self.test_y, class_predict))
        print(classification_report(self.test_y, class_predict))
        print(accuracy_score(self.test_y, class_predict))

    def predict(self, file):
        print('Predict ...')
        data_test = pd.read_csv(file).values
        print(self.classifier.predict(data_test))


if __name__ == '__main__':
    classifier = BillAuthenticationClassifier('../../data/bill_authentication.csv')
    classifier.train()
    classifier.predict('../../data/bill_authentication_test.csv')
