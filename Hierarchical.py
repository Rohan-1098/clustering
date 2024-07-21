import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
Univ = pd.read_csv("../Data/Universities_Clustering.csv")
# print ("Before Normalization")
# print (Univ.head())
# input()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:,1:])
# print ("After Normalization")
# print (df_norm.head())
# input()


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 

# print (type(df_norm))
# input()


#p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")

# plt.figure(figsize=(15, 5));
# plt.title('Hierarchical Clustering Dendrogram for Ai WD Batch');
# plt.xlabel('Index');
# plt.ylabel('Distance')
# sch.dendrogram(
#     z,
#     leaf_rotation=0.,  # rotates the x axis labels
#     leaf_font_size=8.,  # font size for the x axis labels
# )
# plt.show()
# input()	

# help(linkage)
# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from sklearn.cluster import	AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_)
# print (cluster_labels)
# input()
Univ['Cluster_Label']=cluster_labels # creating a  new column and assigning it to new column 
Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]
# print (Univ.head())
# input()

# getting aggregate mean of each cluster
# print (Univ.iloc[:,2:].groupby(Univ.Cluster_Label).mean())
# input()

# creating a csv file 
Univ.to_csv("Univ_clus_WND810.csv",encoding="utf-8")


