# This is my approach to the housing prices data science challenge on Kaggle. 
# Summary: The problem with this prediction is the abundance of information and finding the right features for the model. You could obvioulsy just include all variables 
# which would lead to overfitting. So finding a few strong predictors is the way to go. 
# Generally I want to find time an efficient approach to this task, which will not improve my Kaggle score but I consider very relevant for data science in general. 

# Lets get 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Data import
dfTrain = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
dfTest = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

dfTrain.describe