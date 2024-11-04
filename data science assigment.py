import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.metrics import silhouette_score as ss

data=pd.read_csv(r"C:\Users\aruna\OneDrive\Desktop\sem 2\Data science\prediction data.csv")
print("the data is :")
print(data)
print("NA data")
print(data[["ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"]].isnull())
data1=data.dropna(how="any")
print("after dropping the data is:")
print(data1)
print("data after cleaning:")
print(data1[["ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"]].isnull())

#exploratory data analysis
print("the means of all the values are:")
print(data1[["ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"]].mean())

s1=data1['ALP'].drop(data1.index[(data1['ALP']<44) | (data1['ALP']>147)])
print("series 1\n",s1)
data1['ALP'].plot(color='blue',alpha=0.5)
s1.plot(color="red",alpha=0.5)
plt.xlabel("SIZE")
plt.ylabel('ALP (IU/dL)')
plt.title("ALP outliers")
plt.show()

s2=data1['AST'].drop(data1.index[(data1['AST']<5) | (data1['AST']>40)])
print("series 2:\n",s2)
data1['AST'].plot(color='blue',alpha=0.5)
s2.plot(color="red",alpha=0.5)
plt.xlabel("SIZE")
plt.ylabel('AST (U/dL)')
plt.title("AST outliers")
plt.show()

#NO OUTLIERS IN CHE
s3=data1['CHE'].drop(data1.index[(data1['CHE']<6) | (data1['CHE']>18)])
print("series 3:\n",s3)
data1['CHE'].plot(color='blue',alpha=0.5)
s3.plot(color="red",alpha=0.5)
plt.xlabel("SIZE")
plt.ylabel(' CHE(U/ml  )')
plt.title("CHE outliers")
plt.show()
print("From the plot ,it can be inferred that there are no outliers in CHE factor of the blood")

s4=data1['CHOL'].drop(data1.index[(data1['CHOL']<3.3) | (data1['CHOL']>7.1)])
print("series 4:\n",s4)
data1['CHOL'].plot(color='blue',alpha=0.5)
s4.plot(color="red",alpha=0.5)
plt.xlabel("SIZE")
plt.ylabel(' CHOL RATIO(LDL/HDL)')
plt.title("CHOL outliers")
plt.show()

s5m=data1['CREA'].drop(data1.index[(data1['Sex']=='m') & ((data1['CREA']<65.4) | (data1['CREA']>119.3))])
print("the series 5 with only male data is ",s5m)
data1_crea=data1[['CREA','Sex']]
print(data1_crea)
data1_male=data1_crea[data1_crea['Sex']=='m']
print(data1_male)
data1_male.plot(color="blue",alpha=0.5)
s5m.plot(color="red",alpha=0.5)
plt.xlabel("SIZE")
plt.ylabel(' CREA (micromoles/L)')
plt.title("CREA outliers for male")
plt.show()


s5f=data1['CREA'].drop(data1.index[(data1['Sex']=='f') & ((data1['CREA']<52.2) | (data1['CREA']>91.9))])
print("the series 5 with only female data is ",s5f)
data1_female=data1_crea[data1_crea['Sex']=='m']
print(data1_female)
data1_female.plot(color="blue",alpha=0.5)
s5f.plot(color="red",alpha=0.5)
plt.xlabel("SIZE")
plt.ylabel(' CREA (micromoles/L)')
plt.title("CREA outliers for female")
plt.show()

data_val=data1[["ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"]]
#train test splitting
rows_train=np.random.choice(len(data_val),size=int(0.7*len(data_val)))
train_data=data_val.iloc[rows_train]
print(train_data)

rows_test=np.random.choice(len(data_val),size=int(0.3*len(data_val)))
test_data=data_val.iloc[rows_test]
print(test_data)

data_kmeans=train_data
kmeans=cluster.KMeans(init='random',n_clusters=5)
kmeans.fit(data_kmeans)
labels=kmeans.labels_
print(labels)

pred=kmeans.predict(test_data)
print(pred)
plt.show()

plt.scatter(test_data['ALT'],test_data['GGT'],c=pred)
plt.show()
plt.scatter(test_data['ALP'],test_data['PROT'],c=pred)
plt.show()
inertia=[]
for i in range(1,10):
    kmeans=cluster.KMeans(init='random',n_clusters=i)
    kmeans.fit(train_data)
    inertia.append(kmeans.inertia_)
print(inertia)   
plt.plot(range(1,10),inertia,'ro-')
    
ss1=ss(data_kmeans,labels,metric="euclidean")
print("silhouette score:",ss1)
ss2=ss(test_data,pred,metric="euclidean")
print("silhouette score of prediction",ss2)

