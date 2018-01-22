import pandas as pd

train = pd.read_csv("test.csv")
test  = pd.read_csv("train.csv")

print(test)
print(train)


train_x = train.iloc[:,:-2]
train_y = train['Activity']

test_x = test.iloc[:,:-2]
test_y = test['Activity']
