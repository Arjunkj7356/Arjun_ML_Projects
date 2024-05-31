import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

#importing datasets
data_set = pd.read_csv("C:\\Datascience\\Dataset_Excel\\Customer_Data.csv")
df = pd.DataFrame(data_set)
print("Actual Dataset")
print(df.to_string())
data_set.info()
data_set.isnull().sum()

#Extracting Independent and dependent Variable
x = data_set.iloc[:, [2,3]].values
y = data_set.iloc[:, 4].values

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#feature Scaling
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy")
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred = classifier.predict(x_test)
print("-----------PREDICTION----------")
df2 = pd.DataFrame({"Actual Result-Y":y_test,"PredictionResult":y_pred})
print(df2.to_string())

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))


