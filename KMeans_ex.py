import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data = pd.read_csv("C:\\Datascience\\Dataset_Excel\\Mall_Customers.csv")
df = pd.DataFrame(customer_data)
print(df.to_string())

customer_data.info()

customer_data.isnull().sum()

#Extracting Independent Variables
x = customer_data.iloc[:, [3, 4]].values
print(x)

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_predict = kmeans.fit_predict(x)
print(y_predict)




