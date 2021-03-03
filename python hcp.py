#################### Problem 1 ######################

import pandas as pd
import matplotlib.pyplot as plt

airlines_data = pd.read_excel("F:/assignment/Clustering-Hierarchical Clustering/Dataset_Assignment Clustering/EastWestAirlines.xlsx" , sheet_name = "data")
# airline = drope.airlines_data([ "nhd"] ,axis=1 , inplace = true)

def norm_fun(i):
    x=(i-i.min()) / (i.max() - i.min())
    return (x)

airlines_norm = norm_fun(airlines_data.iloc[:,:])

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

linkage_complet = linkage(airlines_norm , method="complete" , metric="euclidean")
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlabel("index");plt.ylabel('distance')
sch.dendrogram(linkage_complet , leaf_rotation = 0 , leaf_font_size = 10)
plt.show()

linkage_average = linkage(airlines_norm, method='average' , metric='euclidean')
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlabel('index');plt.ylabel('distance')
sch.dendrogram(linkage_average , leaf_rotation=0 , leaf_font_size= 10 )
plt.show()

linkage_centroid = linkage(airlines_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlable('index');plt.ylabel('distance')
sch.dendrogram(linkage_centroid , leaf_rotation=0 , leaf_font_size=10)
plt.show()

linkage_single = linkage(airlines_norm, method="complete", metric="euclidean")
plt.figure(figsize=(15,8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlable('index');plt.ylabel('distance')
sch.dendrogram(linkage_single , leaf_rotation=0 , leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering

sk_linkage_complete = AgglomerativeClustering(n_clusters=3 , affinity="euclidean" , linkage="complete").fit(airlines_norm)
cluster_airlines_complete =pd.Series(sk_linkage_complete.labels_)
airlines_data["cluster"] = cluster_airlines_complete

sk_linkage_single = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean" , linkage="single").fit(airlines_norm)
cluster_airline_single = pd.Series(sk_linkage_single.labels_)
airlines_data["cluster"] = cluster_airline_single

sk_linkage_average = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean" , linkage="average").fit(airlines_norm)
cluster_airline_average = pd.Series(sk_linkage_average.labels_)
airlines_data["cluster"] = cluster_airline_average

sk_linkage_centroid = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean" , linkage="centroid").fit(airlines_norm)
cluster_airline_centroid = pd.Series(sk_linkage_centroid.labels_)
airlines_data["cluster"] = cluster_airline_centroid

airlines = airlines_data.iloc[ : , [12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines_data.iloc[: , 0:].groupby(airlines.cluster).mean()
airlines_data.to_csv("new_airlines",encoding="utf-8")
import os
os.getcwd()

#################### problem2 #####################
import pandas as pd
from scipy.cluster.hierarchy import cluster as sch
import matplotlib.pyplot as plt
from sklearn.cluster import  AgglomerativeClustering
import os

crime_data = pd.read_csv("C:/Users/hp/Desktop/crime_data.csv")

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

crime_data_norm = norm_func(crime_data.iloc[: , 1:])

crime_single_linkage = linkage(crime_data_norm , method="single" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_single_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

crime_complete_linkage = linkage(crime_data_norm , method="complete" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_complete_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

crime_average_linkage = linkage(crime_data_norm , method="average" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using average linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_average_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

crime_centroid_linkage = linkage(crime_data_norm , method="centroid" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using centroid linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(crime_centroid_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

crime_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_complete = pd.Series(crime_complete.labels_)
crime_data["cluster"] = cluster_crime_complete

crime_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_single = pd.Series(crime_single.labels_)
crime_data["cluster"]  = cluster_crime_single

crime_average = AgglomerativeClustering(n_clusters=3 , linkage="average" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_average = pd.Series(crime_average.labels_)
crime_data["cluster"] = cluster_crime_average

crime_centroid = AgglomerativeClustering(n_clusters=3 , linkage="centroid" , affinity="euclidean").fit(crime_data_norm)
cluster_crime_centroid = pd.Series(crime_centroid)
crime_data["cluster"] = cluster_crime_centroid

crime_data = crime_data.iloc[: , [5 , 0 , 1 , 2 , 3 , 4]]
crime_data.iloc[: , 1:].groupby(crime_data.cluster).mean()

crime_data.to_csv("final_crime_data.csv" , encoding="utf-8")

os.getcwd()

############################## Problem3 ###############################################

import pandas as pd
import matplotlib.pyplot as plt


teleco_data = pd.read_excel("F:/assignment/Clustering-Hierarchical Clustering/Dataset_Assignment Clustering/Telco_customer_churn.xlsx")
teleco_data.drop(['Count','Quarter'] , axis=1 , inplace=True)

new_teleco_data = pd.get_dummies(teleco_data)

from sklearn.preprocessing import OneHotEncoder

ohot_enc =OneHotEncoder()

new_teleco_data2 = pd.DataFrame(ohot_enc.fit_transform(teleco_data).toarray()) 






















