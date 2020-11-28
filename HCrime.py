import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df=pd.read_csv("crime_data.csv")
dataframe= df.iloc[:,1:5]
print(dataframe)
print(df.shape)
print(df.head())


minmaxscalar=preprocessing.MinMaxScaler()
x_scaled = minmaxscalar.fit_transform(dataframe)
df_normalized = pd.DataFrame(x_scaled)
print("-----df normalised ")
print(df_normalized.head())

#build the HCluster

import scipy.cluster.hierarchy as sch

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
Z=sch.linkage(df_normalized,method='complete', metric='euclidean')
sch.dendrogram(Z)
plt.show


from sklearn.cluster import AgglomerativeClustering


h_complete=AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(df_normalized)

cluster_labels=pd.Series(h_complete.labels_)
print(cluster_labels)
plt.figure(figsize=(10, 7))
print(df_normalized.iloc[:,0])
print(df_normalized.iloc[:,1:3])
plt.scatter(df_normalized.iloc[:,0],df_normalized.iloc[:,1:2], c=h_complete.labels_, cmap='rainbow')
#plt.show()

df['Crime_cluster']= cluster_labels
print(df.shape)

#df= df[:,[5,0,1,2,3,4]]
print(df.head)

