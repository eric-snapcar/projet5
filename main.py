
# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def loadData():
    test = pd.read_csv("test.csv")
    train  = pd.read_csv("train.csv")
    return test, train

def loadData_():
    test, train = loadData()
    train_x = train.iloc[:,:-2]
    train_y = train['Activity']
    test_x = test.iloc[:,:-2]
    test_y = test['Activity']
    return train_x, train_y, test_x, test_y
def predict(train_x, train_y, test_x, test_y ):
    rfc = RandomForestClassifier(n_estimators=500)
    model = rfc.fit(train_x, train_y)
    return rfc.predict(test_x)
# Les deux dernières colonnes sont subject et Activity
# Les autres colonnes sont celle où il y a des données intéressantes

train_x, train_y, test_x, test_y = loadData_()
pred = predict(train_x, train_y, test_x, test_y)
print accuracy_score(test_y, pred)
