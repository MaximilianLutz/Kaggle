# This code is from the Titanic Competition on Kaggle. Spoiler: I got a score of 98.43, which I'm pretty satisfied with

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import dfTrain_dfTest_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xg

import seaborn as sns
sns.set(style="ticks")

# Never forget: Import Data 
dfTrain = pd.read_csv("dfTrain.csv")
dfTest = pd.read_csv("dfTest.csv")
gd = pd.read_csv("gender_submission.csv")
both = [dfTrain, dfTest]

dfTrain.sample(10)

# Missings?
print('dfTrain data set columns with null values:\n', dfTrain.isnull().sum())
print('dfTest data set columns with null values:\n', dfTest.isnull().sum())

# Lots of nulls for Cabin. I guess this will not be included later, so no na removal. 
# It would be a shame to lose age as explanatory variable. Lets impute the missings: 
for df in both:
    mean_age = round(df['Age'].mean(), 1)
    print(mean_age)
    df['Age'].replace(np.nan, mean_age, inplace=True)
    df = df.drop(['Ticket', 'Cabin'], axis=1)

dfTrain.describe()


# Recoding to numeric: 
for df in both:
    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1

    df.loc[df["Embarked"] == "S", "Embarked"] = 0
    df.loc[df["Embarked"] == "Q", "Embarked"] = 1
    df.loc[df["Embarked"] == "C", "Embarked"] = 2

#The factors of survival on the MS Titanic are rather well known. You are likely to survive, if you are 
#* Female 
#* A child 
#* Not poor 
#* Not the captain

#However, browsing through the data set gives you hints about further explaining variables. Such as: 
#* Harbour of embarkment (I assume this is correlated to the location of the room on the ship. Also might be a proxy for wealth? )
#* Member of a family (Because families where allowed on rescue vessels as a whole)

#On the wild side I guess that these could also look at these: 
#* Room number (Because some rooms where closer to floor exits, or less affected by entering water. However, most of the rows are missing, so maybe we'll just leave this out)
#* Length of name, as another proxy for wealth and an indicator of royal heritage, which might be an additional factor to wealth)   
#* Titles should be an issue. 

# Generate an identifier for children 
for df in both: 
    df['child'] = df['Age'].apply(lambda x: 'True' if x <= 17 else 'False')
    df['child'].value_counts()

# Combine Parents and children for familiy size
    df["FamSize"] = df["SibSp"] + df["Parch"] + 1


for df in both: 
    for name in df["Name"]:
            df["Title"] = df["Name"].str.extract("([A-Za-z]+)\.",expand=True)

    othertitles = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
                  "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other", "Rev": "Other", "Dr": "Other"}    

    df.replace({"Title": othertitles}, inplace=True)


    df.loc[df["Title"] == "Miss", "Title"] = 0
    df.loc[df["Title"] == "Mr", "Title"] = 1
    df.loc[df["Title"] == "Mrs", "Title"] = 2
    df.loc[df["Title"] == "Master", "Title"] = 3
    df.loc[df["Title"] == "Other", "Title"] = 4
    print(set(df["Title"]))
    # No more use for name variable
    df = df.drop(['Name'], axis=1)


# Visual inspection is best inspection

sns.catplot(x="child", y="Survived", kind="bar", data=dfTrain);

sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=dfTrain);

# Wealth
sns.catplot(x="Pclass", y="Survived", kind = "bar", data=dfTrain);

# Fare should be very precise. Quick look: 
dfTrain['Survived'].groupby(pd.qcut(dfTrain['Fare'], 3)).mean()

sns.pointplot(x="Title", y="Survived",  data=dfTrain);

sns.catplot(x="Embarked", y="Survived", kind="bar", data=dfTrain);
print(set(dfTrain["Embarked"]))

sns.pointplot(x="FamSize", y="Survived" , data=dfTrain)


### What did we learn? (mostly obvious) 
#* Women have better chances of survival than men 
#* Children are safer than grown-ups 
#* Being rich definitely helps 
#* Getting on the ship later increases your chance of survival. My assumption is that later boarding translates to cabins closer to exits. 
#* If people approach you as "Mr" you can pretty much jump off the boat. 
#* Single travelling passengers and families bigger than 4 have lower chance of survival. 


### Predictions 
dfTrain = dfTrain.drop(["Name", "Cabin", "Ticket", "Embarked", "child"], axis=1)
dfTest = dfTest.drop(["Name", "Cabin", "Ticket", "Embarked", "child"], axis=1)

from sklearn.linear_model import LogisticRegression

xTrain = dfTrain.drop(["Survived", "PassengerId"], axis=1)
yTrain = dfTrain["Survived"]
xTest  = dfTest.drop("PassengerId", axis=1).copy()
xTrain.shape, yTrain.shape, xTest.shape


logreg = LogisticRegression()
logreg.fit(xTrain, yTrain)
Y_pred = logreg.predict(xTest)
acc_log = round(logreg.score(xTrain, yTrain) * 100, 2)
acc_log

coeff_df = pd.DataFrame(dfTrain.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines

svc = SVC()
svc.fit(xTrain, yTrain)
Y_pred = svc.predict(xTest)
acc_svc = round(svc.score(xTrain, yTrain) * 100, 2)
acc_svc

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(xTrain, yTrain)
Y_pred = gaussian.predict(xTest)
acc_gaussian = round(gaussian.score(xTrain, yTrain) * 100, 2)
acc_gaussian


# Decision Treea

decision_tree = DecisionTreeClassifier()
decision_tree.fit(xTrain, yTrain)
Y_pred = decision_tree.predict(xTest)
acc_decision_tree = round(decision_tree.score(xTrain, yTrain) * 100, 2)
acc_decision_tree

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(xTrain, yTrain)
Y_pred = random_forest.predict(xTest)
random_forest.score(xTrain, yTrain)
acc_random_forest = round(random_forest.score(xTrain, yTrain) * 100, 2)
acc_random_forest


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',
              'Decision Tree'],
    'Score': [acc_svc,acc_log, 
              acc_random_forest, acc_gaussian, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


submission = pd.DataFrame({
        "PassengerId": dfTest["PassengerId"],
        "Survived": Y_pred
    })

# Irrelevant but for completeness:
#submission.to_csv('titanic.csv', index=False)