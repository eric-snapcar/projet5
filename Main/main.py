import pandas as pd
data = pd.read_excel('Online Retail.xlsx')
"""
def total_price(x):
    return x['UnitPrice'] * x['Quantity']
data["Total"]=data.apply(total_price, axis=1) # axis = 1
"""
data_['Total'] = data_['UnitPrice'].multiply(data_['Quantity'])
print data
