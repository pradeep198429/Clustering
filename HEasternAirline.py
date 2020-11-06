import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing

import matplotlib.pyplot as plt

Df=pd.read_csv("EastWestAirlines.csv")


minmaxscalar=preprocessing.MinMaxScaler()
x_scaled = minmaxscalar.fit_transform(Df)
df_normalized = pd.DataFrame(x_scaled)
print(df_normalized)



#build dendogram

import scipy.cluster.hierarchy as sch


plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend= sch.dendrogram(sch.linkage(df_normalized,method='complete', metric='euclidean'))
plt.show




from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=9,	linkage='complete',affinity = "euclidean").fit(df_normalized)

cluster_labels=pd.Series(h_complete.labels_)
print(cluster_labels)
plt.figure(figsize=(10, 7))
plt.scatter(df_normalized.iloc[:,0],df_normalized.iloc[:,8], c=h_complete.labels_, cmap='rainbow')
plt.show()
