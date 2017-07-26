# stock_learning_predictor

a simple python scripts that use knn and random forest classifier as its algorithms for predicting if the particular stock ticker could go up 10%+ in price or not based on its ratios such as PE, PPS, PPB, Div Yield. So the script would not be working properly (won't run) if the ratios are not complete.
it's paired with read_investing.py script that parses through the investing.com website to obtain the ratios needed. But not every ticker's ratios could be obtain this way because of the URL regex is not yet properly set up. 
Just clone the this repository and simply run the script on IPython console.


