import pandas as pd
import numpy as np
from sklearn import metrics

#importing datasets
diabetics = pd.read_csv("C:\\Datascience\\Dataset_Excel\\diabetes2.csv")
df = pd.DataFrame(diabetics)
print(df.to_string())
diabetics.info()
diabetics.isnull().sum()

x = diabetics.iloc[:, :-1].values
y = diabetics.iloc[:, 8].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result:
y_pred = regressor.predict(x_test)

from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
# fit the model with data
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
df2 = pd.DataFrame(x_test)
#test data
print(df2.to_string())
#pred. data
print(y_pred)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))