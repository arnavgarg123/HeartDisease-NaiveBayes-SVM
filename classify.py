import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
linreg = LinearRegression()
# importing the data from csv files
# cd ~
# cd Documents
# cd GitHub
# cd NaiveBayes - HeartDisease/
df = pd.read_csv(
    '~/Documents/GitHub/NaiveBayes-HeartDisease/tubes2_HeartDisease_test.csv', na_values=['?'])
df1 = pd.read_csv(
    '~/Documents/GitHub/NaiveBayes-HeartDisease/tubes2_HeartDisease_train.csv', na_values=['?'])
df.shape
df1
df1.shape
df1.isnull().sum()

df1.dropna(thresh=8, inplace=True)
df1.isnull().sum()
'''dfn=df.dropna()
dfn.head()
train_x=dfn.iloc[:,1:]
train_y=dfn.iloc[:,:1]
linreg.fit(train_x,train_y)
test_data=df.iloc[:,1:]
age_prediction['Column1']=pd.DataFrame(linreg.predict(test_data))
'''
df1['Column5'].replace(0, 221, inplace=True)
##########        Data Cleaning        ##########
# median to fill missing values
df1['Column4'].fillna(df['Column4'].median(), inplace=True)
df1['Column5'].fillna(df['Column4'].median(), inplace=True)
df1['Column7'].fillna(df['Column7'].max(), inplace=True)
df1['Column8'].fillna(df['Column8'].mean(), inplace=True)
df1['Column9'].fillna(df['Column9'].max(), inplace=True)
df1['Column10'].fillna(df['Column10'].mean(), inplace=True)
df1.isnull().sum()
df[df['Column4'].isnull()]

dfn = df1.dropna()
dfn.head()
train_x = dfn.drop(['Column6'], axis=1).iloc[:18, [3, 6]]
train_y = dfn.iloc[:18, 11]
linreg.fit(train_x, train_y)
test_data = df1.drop(['Column6'], axis=1).iloc[:18, [3, 6]]
a = pd.DataFrame(linreg.predict(test_data))
a.head(18)
