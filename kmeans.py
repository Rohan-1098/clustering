import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

# Generating random uniform numbers 
X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)

# print (X)
# print (len(X))
# # print (Y)
# input()
df_xy =pd.DataFrame(columns=["X","Y"])
# print (df_xy.head())
# input()

df_xy.X = X
df_xy.Y = Y
# print (df_xy.head())
# input()
# df_xy.plot(x="X",y = "Y",kind="scatter")
# plt.show()
# input()
model1 = KMeans(n_clusters=3).fit(df_xy)

# print (model1.labels_)
# input()

# df_xy.plot(x="X",y = "Y",
# 	c=model1.labels_,kind="scatter",
# 	s=10,cmap=plt.cm.coolwarm)
# plt.show()
# input()


# Kmeans on University Data set 
Univ = pd.read_csv("../Data/Universities_Clustering.csv")

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:,1:])


# print (df_norm.head(10))  # Top 10 rows
# input()

###### screw plot or elbow curve ############
k = list(range(2,20))
# print (k)
# input()
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Scree plot 
plt.plot(k,TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS");
plt.xticks(k)
plt.show()
input()

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model_kmeans = KMeans(n_clusters=4) 
model_kmeans.fit(df_norm)

# print(model)
# input()
# input() # getting the labels of clusters assigned to each row 
md=pd.Series(model_kmeans.labels_)  # converting numpy array into pandas series object 
Univ['Cluster_Label_K_Means'] = md # creating a  new column and assigning it to new column 
# df_norm.head()
# df_norm['clust'] = md
# print (Univ)
# input()
# print (df_norm)
# input()
print (Univ.iloc[:,1:7].groupby(Univ.Cluster_Label_K_Means).mean())
# input()

Univ.to_csv("Univ_clus_WND26AI_K_means_4.csv",encoding="utf-8")
