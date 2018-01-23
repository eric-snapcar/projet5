
# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
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

def filter(train_x, test_x, model ):
    select = SelectFromModel(model, prefit=True, threshold=0.005)
    train_x_filtered = select.transform(train_x)
    test_x_filtered = select.transform(test_x)
    return train_x_filtered,test_x_filtered

def rate(test_y,pred ):
    print accuracy_score(test_y, pred)

def predict(train_x, test_x, train_y):
    rfc = RandomForestClassifier(n_estimators=500)
    model = rfc.fit(train_x, train_y)
    return rfc.predict(test_x), model

# Les deux dernières colonnes sont subject et Activity
# Les autres colonnes sont celle où il y a des données intéressantes

train_x, train_y, test_x, test_y = loadData_()
pred, model = predict(train_x, test_x, train_y)
print("Unfiltered accuracy")
rate(test_y, pred)

train_x_filtered,test_x_filtered = filter(train_x, test_x, model )
pred_filtered = predict(train_x_filtered, test_x_filtered, train_y )[0]
print("Filtered accuracy")
rate(test_y, pred_filtered)
