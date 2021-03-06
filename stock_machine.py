# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:00:27 2017

@author: Bana
"""
import pandas
from yahoo_finance import Share
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
import numpy as np
import sys
from termcolor import colored,cprint
from investing_read_stock import investing_read
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
        
forest =  RandomForestClassifier(n_estimators=5, random_state=2)  
data=data.values
target = target.values.reshape(len(target),)
forest.fit(data,target)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gint', max_depth=None, max_features='auto', max_leaf_nodes=None,min_samples_leaf=1,min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=5,n_jobs=1,oob_score=False, random_state=2,verbose= 2, warm_start=False)
y_pred_frt = forest.predict(data)
print("Random Forest Prediction accuracy based on training: %.2f" %(np.mean(y_pred_frt == target)))

#KNN initialization        
knn =  KNeighborsClassifier(n_neighbors=1)  
knn.fit(data,target)
KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=1,p=2,weights='uniform')
y_pred_knn = knn.predict(data)
print("KNN prediction accuracy based on training: %.2f" %(np.mean(y_pred_knn == target)))

#Perceptron Initialization
ppn = Perceptron(n_iter=40, eta0=0.1, random_state =0 )
ppn.fit(data,target)
y_pred_ppn = ppn.predict(data)
print("PPN prediction accuracy based on training: %.2f" %(np.mean(y_pred_ppn == target)))

    
"""
to make prediction input (eps,book_prc,peg,pe) of a stock, 1 means more than
10% increase over 200 days
"""
def main():
    
    trigger = input("input S for inputing stock ticker or T to manually input price per sales, price per book, dividend yield, pe ratio: ")
    if trigger == 'S':
        stock_ticker = input("input stock ticker: ")
        s = Share(stock_ticker)
        try:
            name = s.get_name()
            pps = float(s.get_price_sales())
            book_prc = float(s.get_price_book())
            divy = float(s.get_dividend_yield())
            pe = float(s.get_price_earnings_ratio())   
        
        except :
            saham = investing_read(stock_ticker)
            try:
                saham.get_text()
            except:
                print("url not found, provide investing.com ratios url \n")
                saham.manual_text()
            try:    
                pps,book_prc,divy,pe = saham.read_text()
            except ValueError:
                print("data is not complete cannot begin prediction...\n")
                sys.exit(0)
                
        x_new = np.array([[pps,book_prc,divy,pe]])
        
    elif trigger == 'T':
        stock_ticker = input("input stock ticker: \r")
        s = Share(stock_ticker)
        name = s.get_name()
        args = map(float,input("input argument (pps,book_prc,divy,pe): \r").split(" "))
        pps,book_prc,divy,pe = list(args)
        x_new = np.array([[pps,book_prc,divy,pe]])
        
    else:
        print("Input S or T \n")
        sys.exit(0)
        
    prediction_frt = forest.predict(x_new)
    prediction_knn = knn.predict(x_new)
    if prediction_frt == 1:
        print(colored("%s, it'll go up 10%%+ based on rnd forest" %(name),'green'))
    else:
        print(colored("%s, it'll probably become a loser based on rnd forest" %(name),'red'))    
        
    if prediction_knn == 1:
        print(colored("%s, it'll go up 10%%+ based on KNN" %(name),'green'))
    else:
        print(colored("%s, it'll probably become a loser based on KNN" %(name),'red'))    

if __name__ == '__main__':
    main()

