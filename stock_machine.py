# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:00:27 2017

@author: lenovo
"""
import pandas
from yahoo_finance import Share
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys
#unquote to get stock data
"""
f = open("SP100.txt","r")
dow = [x.strip("\n") for x in f.readlines()]

data_spek = [[] for i in range(len(dow))]
target = []

for i in range(len(dow)):
    s = Share(dow[i])
    
    try:
        eps = float(s.get_EPS_estimate_current_year())
        book_prc = float(s.get_price_book())
        peg = float(s.get_price_earnings_growth_ratio())
        pe = float(s.get_price_earnings_ratio())   
        chg = s.get_percent_change_from_200_day_moving_average()
        chg = float(chg.replace("%",""))
        if chg>10:
            target.append(1)
        else:
            target.append(0)
        data_spek[i]=[eps,book_prc,peg,pe]     

    except:
        target.append(0)
        pass        




df = pandas.DataFrame(data=data_spek)
df.to_clipboard(excel=True)
"""

data = pandas.read_csv('data_spek.csv')
target = pandas.read_csv('target_spek.csv')

#unquote to see the classification
"""
fig, ax = plt.subplots(3,3, figsize=(15,15))

for i in range(3):
    for j in range(3):
        ax[i,j].scatter(data.iloc[:,j],data.iloc[:,i+1],c=target,s=60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        if j>i:
            ax[i,j].set_visible(False)
"""    
        
knn =  KNeighborsClassifier(n_neighbors=1)  
data=data.values
target = target.values.reshape(len(target),)
knn.fit(data,target)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=1,p=2,weights='uniform')
         
y_pred = knn.predict(data)
print("Accuracy: %.2f" %(np.mean(y_pred == target)))

"""
to make prediction input (eps,book_prc,peg,pe) of a stock, 1 means more than
10% increase over 200 days
"""
def main():
    
    stock_ticker = input("input stock ticker: ")
    s = Share(stock_ticker)
    try:
        eps = float(s.get_EPS_estimate_current_year())
        book_prc = float(s.get_price_book())
        peg = float(s.get_price_earnings_growth_ratio())
        pe = float(s.get_price_earnings_ratio())   
    except:
        print ("the data is not complete, select another stock")
        sys.exit(0)
    x_new = np.array([[eps,book_prc,peg,pe]])
    prediction = knn.predict(x_new)
    if prediction == 1:
        print ("the stock will go up 10%+")
    else:
        print("it'll probably become a junk")    

if __name__ == '__main__':
    main()

