import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import scipy.cluster.hierarchy as sch

df=pd.read_csv('F:/Explo Data/London Data/daily_dataset.csv/daily_dataset.csv', parse_dates=["day"],index_col ="day")
df.reset_index(inplace=True)
piv=df.pivot(index='day',columns='LCLid', values='energy_sum')

piv.reset_index(inplace=True)
from datetime import datetime, date
piv['Year-Month']=piv['day'].dt.strftime('%Y-%m')

gr=piv.groupby('Year-Month').mean()

result = gr.transpose() 

ipl=result.interpolate(method='linear', limit_direction='both', axis=1)

new_data = ipl.dropna(axis = 0, how ='any') 

from sklearn import preprocessing

normalized_data = preprocessing.normalize(new_data)

from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score
X=pd.DataFrame(normalized_data)

X_scaled = X
km_silhouette = []
vmeasure_score =[]
db_score = []
for i in range(2,12):
    hc = AgglomerativeClustering(n_clusters = i, affinity = 'euclidean', linkage ='ward')
    preds=hc.fit_predict(X_scaled)
    #print("Score for number of cluster(s) {}: {}".format(i,km.score(X_scaled)))
    #km_scores.append(-km.score(X_scaled))
    
    silhouette = silhouette_score(X_scaled,preds)
    km_silhouette.append(silhouette)
    print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))
    
    db = davies_bouldin_score(X_scaled,preds)
    db_score.append(db)
    print("Davies Bouldin score for number of cluster(s) {}: {}".format(i,db))

    print("-"*100)
    
plt.figure(figsize=(7,4))
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,12)],y=km_silhouette,s=150,edgecolor='k')
plt.grid(True)plt.figure(figsize=(7,4))
plt.title("The silhouette coefficient method \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,12)],y=km_silhouette,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(2,12)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Silhouette score",fontsize=15)
plt.xticks([i for i in range(2,12)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()

plt.scatter(x=[i for i in range(2,12)],y=db_score,s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Davies-Bouldin score")
plt.show()

from sklearn.mixture import GaussianMixture

gm_bic= []
gm_score=[]
for i in range(2,12):
    gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(X_scaled)
    print("BIC for number of cluster(s) {}: {}".format(i,gm.bic(X_scaled)))
    print("Log-likelihood score for number of cluster(s) {}: {}".format(i,gm.score(X_scaled)))
    print("-"*100)
    gm_bic.append(-gm.bic(X_scaled))
    gm_score.append(gm.score(X_scaled))

plt.figure(figsize=(7,4))
plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
plt.scatter(x=[i for i in range(2,12)],y=np.log(gm_bic),s=150,edgecolor='k')
plt.grid(True)
plt.xlabel("Number of clusters",fontsize=14)
plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
plt.xticks([i for i in range(2,12)],fontsize=14)
plt.yticks(fontsize=15)
plt.show()

plt.scatter(x=[i for i in range(2,12)],y=gm_score,s=150,edgecolor='k')
plt.show()
