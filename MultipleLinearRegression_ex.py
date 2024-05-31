import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
data = pd.read_csv("C:\\Datascience\\Dataset_Excel\\Fuel_Consumption_2000-2022.csv")
print(data)
data.info()
data.isnull().sum()

a = data.drop(["HWY (L/100 km)", "COMB (L/100 km)", "FUEL"],axis=1)
print(a)

#Extracting Independent and dependent Variable
x = data.iloc[:, :-1].values
y = data.iloc[:, 10].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x= LabelEncoder()

x[:, 1]= labelencoder_x.fit_transform(x[:,1])
x[:, 2]= labelencoder_x.fit_transform(x[:,2])
x[:, 3]= labelencoder_x.fit_transform(x[:,3])
x[:, 6]= labelencoder_x.fit_transform(x[:,6])
x[:, 7]= labelencoder_x.fit_transform(x[:,7])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result;
y_pred = regressor.predict(x_test)

#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())

print("Mean")
print(data.describe())
print("-------------------------------------")

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score

# predicting the accuracy score
score = r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")


