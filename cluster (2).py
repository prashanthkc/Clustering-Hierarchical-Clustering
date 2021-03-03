import pandas as pd
import matplotlib.pyplot as plt

airlines = pd.read_excel("C:/Users/hp/Desktop/EastWestAirlines.xlsx" , sheet_name="data")
#airlines.drop(['ID#'] , axis=1 , inplace=True)

def norm_fun(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

airlines_norm = norm_fun(airlines.iloc[:,0:])

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

linkage_complete = linkage(airlines_norm , method="complete" ,metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram for complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_complete , leaf_rotation= 0 , leaf_font_size= 10)
plt.show()

linkage_average = linkage(airlines_norm , method="average" ,metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram for average linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_average , leaf_rotation= 0 , leaf_font_size= 10)
plt.show()

linkage_single = linkage(airlines_norm , method="single" ,metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram for single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_single , leaf_rotation= 0 , leaf_font_size= 10)
plt.show()

linkage_centroid = linkage(airlines_norm , method="centroid" ,metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram for centroid linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage_centroid , leaf_rotation= 0 , leaf_font_size= 10)
plt.show()

from sklearn.cluster import AgglomerativeClustering

sk_linkage_complete = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean" , linkage="complete").fit(airlines_norm)
cluster_airline_complete = pd.Series(sk_linkage_complete.labels_)
airlines["cluster"] = cluster_airline_complete

sk_linkage_single = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean" , linkage="single").fit(airlines_norm)
cluster_airline_single = pd.Series(sk_linkage_single.labels_)
airlines["cluster"] = cluster_airline_single

sk_linkage_average = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean" , linkage="average").fit(airlines_norm)
cluster_airline_average = pd.Series(sk_linkage_average.labels_)
airlines["cluster"] = cluster_airline_average

sk_linkage_centroid = AgglomerativeClustering(n_clusters=3 ,affinity="euclidean" , linkage="centroid").fit(airlines_norm)
cluster_airline_centroid = pd.Series(sk_linkage_centroid.labels_)
airlines["cluster"] = cluster_airline_centroid

airlines = airlines.iloc[: , [12 , 0 ,1 , 2, 3, 4 , 5, 6, 7, 8, 9, 10 ,11]]

airlines.iloc[: , 0:].groupby(airlines.cluster).mean()
airlines.to_csv("new_airlines" ,encoding="utf-8")

import os
os.getcwd()


####################################Problem 2#################################################################################
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
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

import os

crime_data.to_csv("final_crime_data.csv" , encoding="utf-8")

os.getcwd()

################################Problem 3#############################################
import pandas as pd

telco_data = pd.read_excel("C:/Users/hp/Desktop/Telco_customer_churn.xlsx")
telco_data.drop(['Count' , 'Quarter'] , axis=1 , inplace=True)

new_telco_data = pd.get_dummies(telco_data)

from sklearn.preprocessing import  OneHotEncoder

OH_enc = OneHotEncoder()

new_telco_data2 = pd.DataFrame(OH_enc.fit_transform(telco_data).toarray())

from sklearn.preprocessing import  LabelEncoder
L_enc = LabelEncoder()
telco_data['Referred a Friend'] = L_enc.fit_transform(telco_data['Referred a Friend'])
telco_data['Offer'] = L_enc.fit_transform(telco_data['Offer'])
telco_data['Phone Service'] = L_enc.fit_transform(telco_data['Phone Service'])
telco_data['Multiple Lines'] = L_enc.fit_transform(telco_data['Multiple Lines'])
telco_data['Internet Service'] = L_enc.fit_transform(telco_data['Internet Service'])
telco_data['Internet Type'] = L_enc.fit_transform(telco_data['Internet Type'])
telco_data['Online Security'] = L_enc.fit_transform(telco_data['Online Security'])
telco_data['Online Backup'] = L_enc.fit_transform(telco_data['Online Backup'])
telco_data['Device Protection Plan'] = L_enc.fit_transform(telco_data['Device Protection Plan'])
telco_data['Premium Tech Support'] = L_enc.fit_transform(telco_data['Premium Tech Support'])
telco_data['Streaming TV'] = L_enc.fit_transform(telco_data['Streaming TV'])
telco_data['Streaming Movies'] = L_enc.fit_transform(telco_data['Streaming Movies'])
telco_data['Streaming Music'] = L_enc.fit_transform(telco_data['Streaming Music'])
telco_data['Unlimited Data'] = L_enc.fit_transform(telco_data['Unlimited Data'])
telco_data['Contract'] = L_enc.fit_transform(telco_data['Contract'])
telco_data['Paperless Billing'] = L_enc.fit_transform(telco_data['Paperless Billing'])
telco_data['Payment Method'] = L_enc.fit_transform(telco_data['Payment Method'])

telco_data.isna().sum()
def std_fun(i):
    x = (i-i.mean()) / (i.std())
    return (x)

telco_data_norm = std_fun(new_telco_data)

str(telco_data_norm)

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage

telco_single_linkage = linkage(telco_data_norm , method="single" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_single_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_complete_linkage = linkage(telco_single_linkage , method="complete" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using complete linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_single_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_average_linkage = linkage(telco_complete_linkage , method="average" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_average_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

telco_centroid_linkage = linkage(telco_data_norm , method="centroid" , metric="euclidean")
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram using single linkage');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(telco_centroid_linkage , leaf_font_size=10 ,leaf_rotation=0)
plt.show()

from sklearn.cluster import  AgglomerativeClustering

telco_single = AgglomerativeClustering(n_clusters=3 , linkage="single" , affinity="euclidean").fit(telco_data_norm)
cluster_telco_single = pd.Series(telco_single.labels_)
telco_data["cluster"] = cluster_telco_single

telco_complete = AgglomerativeClustering(n_clusters=3 , linkage="complete" , affinity="euclidean").fit(telco_data_norm)
cluster_telco_complete = pd.Series(telco_complete.labels_)
telco_data["cluster"] = cluster_telco_complete

telco_average = AgglomerativeClustering(n_clusters=3 , linkage="average" , affinity="euclidean").fit(telco_data_norm)
cluster_telco_average = pd.Series(telco_average.labels_)
telco_data["cluster"] = cluster_telco_average

telco_centroid = AgglomerativeClustering(n_clusters=3 , linkage="centroid" , affinity="euclidean").fit(telco_data_norm)
cluster_telco_centroid = pd.Series(telco_centroid.labels_)
telco_data["cluster"] = cluster_telco_centroid

telco_data.iloc[: , 0:29].groupby(telco_data.cluster).mean()

import os

crime_data.to_csv("final_telco_data.csv" , encoding="utf-8")

os.getcwd()

#############################Program 4##############################################

