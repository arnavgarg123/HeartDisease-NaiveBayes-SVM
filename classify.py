import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
# importing the data from csv files
model = svm.SVC(kernel='rbf', C=1, gamma=1)
df1 = pd.read_csv(
    '~/WorkSpace//GitHub/NaiveBayes-HeartDisease/tubes2_HeartDisease_train.csv', na_values=['?'])
#df1 = pd.read_csv('~/WorkSpace//GitHub/NaiveBayes-HeartDisease/heart.csv', na_values=['?'])
df1.shape
#df1.dropna(thresh=8, inplace=True)
df1.dropna(inplace=True)
#
X = df1[['Column1', 'Column2', 'Column3', 'Column4', 'Column5', 'Column6',
         'Column7', 'Column8', 'Column9', 'Column10', 'Column11', 'Column12', 'Column13']]
y = df1['Column14']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.predict(X_test)
# Plot
sns.distplot(df1['Column14'])
sns.heatmap(df1.corr())
sns.pairplot(df1[['Column11', 'Column10']].dropna())
#

df1.head(10)
df1['Column5'].replace(0, 221, inplace=True)
##########        Data Cleaning        ##########
# median to fill missing values
df1['Column4'].fillna(df1['Column4'].median(), inplace=True)
df1['Column5'].fillna(df1['Column5'].median(), inplace=True)
df1['Column6'].fillna(df1['Column6'].mode(), inplace=True)
df1['Column7'].fillna(df1['Column7'].median(), inplace=True)
df1['Column8'].fillna(df1['Column8'].median(), inplace=True)
df1['Column9'].fillna(df1['Column9'].mode(), inplace=True)
df1['Column10'].fillna(df1['Column10'].median(), inplace=True)
df1[df1['Column11'].isnull()]
df1.isnull().sum()
