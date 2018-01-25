import pandas as pd

def total_price(x):
    return x['UnitPrice'] * x['Quantity']

data = pd.read_excel('Online Retail.xlsx')
data["Total"]=data.apply(total_price)
print data
