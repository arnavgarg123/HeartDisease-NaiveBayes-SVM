from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
# importing the data from csv files
#model = svm.SVC(kernel='rbf', C=1, gamma=1)
df1 = pd.read_csv(
    '~/WorkSpace//GitHub/NaiveBayes-HeartDisease/tubes2_HeartDisease_train.csv', na_values=['?'])
#df1 = pd.read_csv('~/WorkSpace//GitHub/NaiveBayes-HeartDisease/heart.csv', na_values=['?'])
df1.shape
df1.dropna(thresh=8, inplace=True)
df1.dropna(subset=['Column11', 'Column12', 'Column13'], how='all', inplace=True)
#

#model.fit(X_train, y_train)
#model.score(X_train, y_train)
# model.predict(X_test)
# Plot
# sns.distplot(df1['Column14'])
# sns.heatmap(df1.corr())
# sns.pairplot(df1)
#

df1.head(10)
df1['Column5'].replace(0, 221, inplace=True)
##########        Data Cleaning        ##########
# median to fill missing values
df1['Column4'].fillna(df1['Column4'].median(), inplace=True)
df1['Column5'].fillna(df1['Column5'].median(), inplace=True)
df1['Column6'].fillna(0, inplace=True)
df1['Column7'].fillna(df1['Column7'].median(), inplace=True)
df1['Column8'].fillna(df1['Column8'].median(), inplace=True)
df1['Column9'].fillna(0, inplace=True)
df1['Column12'].fillna(0, inplace=True)
df1['Column13'].fillna(3, inplace=True)
df1['Column11'].fillna(2, inplace=True)
df1['Column10'].fillna(df1['Column10'].median(), inplace=True)
df1[df1['Column11'].isnull()]
df1.isnull().sum()
X = df1[['Column3', 'Column9', 'Column11', 'Column12', 'Column13']]
y = df1['Column14']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# df1['Column11'].value_counts()
model = GaussianNB()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

plt.scatter(y_test, prediction)
print(classification_report(y_test, prediction))

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
#
#
#
#
#
#
model = SVC()
model.fit(X_train, y_train)
predictions1 = model.predict(X_test)
predictions1
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
