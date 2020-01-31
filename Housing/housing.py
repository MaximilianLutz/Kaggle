# %% [code]
# This is my approach to the housing prices data science challenge on Kaggle. 
# Summary: The problem with this project is finding the right features for the model. Including all of the many variables 
# would lead to overfitting. So finding a few strong predictors is the way to go. 
# Generally I want to find time an efficient approach to this task, which will not improve my Kaggle score but I consider very relevant for data science in general. 
# I will therefore try to keep this code as short as possbile. So, see this one as the quick and dirty approach, without much theoretical work. 
# Spoiler: My decision tree got a score of 99.45.

# Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Modeling
# Model Prediction: 
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# Data import
dfTrain = pd.read_csv('../input/train.csv')
dfTest = pd.read_csv('../input/test.csv')
both = [dfTrain, dfTest]

# Lets take a quick look at our DF.
dfTrain.sample(15)
# I feel like it does not make too much sense to look at single variables in Detail at this point. Let's take a shortcut and pick variables that strongly correlate with the housing price.

# %% [code]
# So, lets get a good old fashioned list of correlation values. Also, keep the list short. 

#correlation matrix
corrmat = dfTrain.corr()
corrmat.nlargest(16, 'SalePrice', keep = 'all')['SalePrice'][1:]

# %% [code]
# This seems promising. Lets look at these in detail: 
dfTemp = dfTrain.loc[:,corrmat.nlargest(15, 'SalePrice')['SalePrice'].index]
dfTemp.loc[:,'Id']= dfTrain['Id'] #we will need this later. 
dfTemp

# %% [code]
# Until this point we didn't even have to think about contents of our variables. 
# I am going to skip checking for homoskedascity and  outliers and focus on a well specified model. 
# This means further minimizing our feature selection. 
# There are two couples of variables that I expect to be rather similar: 
# Squarefeet of Basement and 1st floor probably strongly corelate 
# as well as GarageCars and GarageArea. Lets take a look: 
dfTemp.corr()

# %% [code]
# Indeed, strong correlation for the two, as well as YearBuilt and GarageYrBlt. 
# Lets remove the ones with weaker correlation on SalePrice 
dfTemp = dfTemp.drop(['GarageYrBlt', '1stFlrSF', 'GarageArea'], axis = 1)

# %% [code]
# After reducing information to get a accessible data frame, let's look at it: 
dfTemp.dtypes

# Only numerical. This makes things easy. Missing values? 
dfTemp.isnull().sum()

# %% [code]
dfTest = dfTest.loc[:,list(dfTemp.columns)]
dfTest = dfTest.drop('SalePrice', axis=1)

# %% [code]
dfTemp = dfTemp.dropna()
dfTest = dfTest.dropna()

# %% [code]
xTrain = dfTemp.drop("SalePrice", axis=1)
xTrain = xTrain.drop('Id', axis=1)
yTrain = dfTemp["SalePrice"]
xTest  = dfTest.drop("Id", axis=1).copy()
xTrain.shape, yTrain.shape, xTest.shape

# %% [code]
linreg = LinearRegression()
linreg.fit(xTrain, yTrain)
Y_pred = linreg.predict(xTest)
acc_lin = round(linreg.score(xTrain, yTrain) * 100, 2)
acc_lin

# %% [code]
# Support Vector Machines

svc = SVC()
svc.fit(xTrain, yTrain)
Y_pred = svc.predict(xTest)
acc_svc = round(svc.score(xTrain, yTrain) * 100, 2)
acc_svc

# %% [code]
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(xTrain, yTrain)
Y_pred = decision_tree.predict(xTest)
acc_decision_tree = round(decision_tree.score(xTrain, yTrain) * 100, 2)
acc_decision_tree

# Cross-validation of models: 
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Linear Regression',  
              'Decision Tree'],
    'Score': [acc_svc, acc_lin, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

# %% [code]
submission = pd.DataFrame({
        "Id": dfTest["Id"],
        "SalePrice": Y_pred
    })
submission.to_csv('housingprice_lutz.csv', index=False)

# %% [code]
# Summary: I skipped theoretical investigation as well as controlling for statistical problems (that are rarely an issue anyways)
# in favor short code and time effective analysis. 
# While the predictions are very precise, contextual analysis should (and can) ussually not be skipped in order to guarantee for 
# valid results. 