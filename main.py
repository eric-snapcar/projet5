
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
def predict(train_x, train_y, test_x ):
    rfc = RandomForestClassifier(n_estimators=500)
    model = rfc.fit(train_x, train_y)
    return rfc.predict(test_x), model
def rate(test_y,pred ):
    print accuracy_score(test_y, pred)
# Les deux dernières colonnes sont subject et Activity
# Les autres colonnes sont celle où il y a des données intéressantes

train_x, train_y, test_x, test_y = loadData_()
pred, model = predict(train_x, train_y, test_x)

from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(model, prefit=True, threshold=0.005)
train_x2 = select.transform(train_x)
print("Train")
print(train_x.shape)
print(train_x2.shape)
test_x2 = select.transform(test_x)
print("Test")
print(test_x.shape)
print(test_x2.shape)
pred_2 = predict(train_x2, train_y, test_x2)[0]
print("Pred")
print(pred.shape)
print(pred_2.shape)
rate(test_y, pred)
rate(test_y, pred_2)
