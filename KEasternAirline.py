import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import  KMeans
from sklearn import preprocessing

df = pd.read_csv("EastWestAirlines.csv")
Preprocess_df=preprocessing.normalize(df)
normalised_df = pd.DataFrame(Preprocess_df)
print(normalised_df)

model= KMeans(n_clusters=9).fit(normalised_df)
plt.figure(figsize=(10, 7))
plt.scatter(normalised_df.iloc[:,0], normalised_df.iloc[:,1], c=model.labels_, cmap='rainbow')
plt.show()
