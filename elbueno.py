# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 09:01:21 2020

@author: Admin
"""


import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


archivo= "cleantweets.csv"
#print(iris.DESCR)

df=pd.read_csv(archivo,names=['sentiment','polarity','subjectivity'])
print(df.head())
pd.set_option('precision',2)
print(df.describe())

X = df.iloc[:,1:3].values
y = df.iloc[:,0].values

print("")
print("K-Means")
kmeans=KMeans(n_clusters=3, random_state=11)

print(kmeans.fit(X))

print("0:50")
print(kmeans.labels_[0:50])
print("50:100")
print(kmeans.labels_[50:100])
print("100:150")
print(kmeans.labels_[100:150])

print("")
print("PCA")

pca=PCA(n_components=2, random_state=11)

pca.fit(X)
datapca=pca.transform(X)
print(pca.fit(X))
print(datapca.shape)

pca_df=pd.DataFrame(datapca,columns=['Component1', 'Component2'])
pca_df['sentiment']=y

axes=sns.scatterplot(data=pca_df, x='Component1',y='Component2',hue='sentiment',legend='brief',palette='cool')

pca_centers=pca.transform(kmeans.cluster_centers_)

dots=plt.scatter(pca_centers[:,0], pca_centers[:,1], s=100, c='k')


