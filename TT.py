import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import decomposition#降纬算法，将64个特征缩减为2个，本质保留方差最大的特征
from itertools import product
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

def dataframe_to_csv(df,csv_name):
    test=pd.DataFrame(df)
    output_path = '/Users/wayne/Desktop/HON'
    test.to_csv(output_path + '/' + csv_name)

def Draw_HON(column,row,first,last,labels,dataframe):
    c = 0
    fig, axs = plt.subplots(row,column,figsize=(15,9),subplot_kw=dict(aspect="equal"))
    cmap = plt.get_cmap("Set3")
    for i in dataframe.index[first:last]:
        score = dataframe.loc[i].values[1:]
        patches,l_text,p_text = axs[c%row,int(c/row)].pie(score.flatten(),
                            explode=(0.1,0.2,0,0,0,0,0,0,0,0.1),
                            pctdistance=0.9,radius=1,
                            colors=cmap(np.arange(10)/10),
                            wedgeprops=dict(width=0.3, edgecolor='k'),
                            autopct='%1.0f%%')

        axs[c%row,int(c/row)].set_title(dataframe.loc[i].values[0],fontsize=20,fontweight='bold')
        for t in l_text:t.set_size(15)
        for t in p_text:t.set_size(15)
        c += 1

    plt.legend(labels, title="Hierarchy of Needs",
              loc="right",
              bbox_to_anchor=(1, 0, 0.5, 1),fontsize=15)
    plt.subplots_adjust(wspace =0.1, hspace =0.1)#调整子图间距
    plt.show()

labels = ["PNs","SNs","BNs","Ens","CNs","Ans","SPNs","SMNs","SANs","TNs"]
df_hons = pd.read_csv('TEST.csv',index_col=[0],parse_dates = True)
#Draw_HON(3,2,0,6,labels,df_hons)
# Draw_HON(3,2,5,11,labels,df_hons)
#Draw_HON(3,2,11,17,labels,df_hons)
#Draw_HON(3,2,17,22,labels,df_hons)
#Draw_HON(3,2,22,28,labels,df_hons)


def Draw_Score(column,row,first,last,dataframe):
    games = dataframe.index.values
    items = dataframe.columns.values
    cmap = plt.get_cmap("Set3")
    fig, axs = plt.subplots(row, column, figsize=(15,9), sharey=True)
    c = 0
    for i in games[first:last]:
        score = dataframe.loc[i].values
        axs[c%row,int(c/row)].bar(range(1,4), score[:3],label='PNs',edgecolor='k',color=cmap(0))
        axs[c%row,int(c/row)].bar(range(4,6), score[3:5],label='SNs',edgecolor='k',color=cmap(1))
        axs[c%row,int(c/row)].bar(range(6,10), score[5:9],label='BNs',edgecolor='k',color=cmap(2))
        axs[c%row,int(c/row)].bar(range(10,15), score[9:14],label='ENs',edgecolor='k',color=cmap(3))
        axs[c%row,int(c/row)].bar(range(15,24), score[14:23],label='CNs',edgecolor='k',color=cmap(4))
        axs[c%row,int(c/row)].bar(range(24,31), score[23:30],label='ANs',edgecolor='k',color=cmap(5))
        axs[c%row,int(c/row)].bar(range(31,33), score[30:32],label='SPNs',edgecolor='k',color=cmap(6))
        axs[c%row,int(c/row)].bar(range(33,35), score[32:34],label='SMNs',edgecolor='k',color=cmap(7))
        axs[c%row,int(c/row)].bar(range(35,41), score[34:40],label='SANs',edgecolor='k',color=cmap(8))
        axs[c%row,int(c/row)].bar(range(41,43), score[40:42],label='TNs',edgecolor='k',color=cmap(9))
        axs[c%row,int(c/row)].set_title(i,fontsize=20,fontweight='bold')
        axs[c%row,int(c/row)].set_xlabel('Items',fontsize=15)
        axs[c%row,int(c/row)].set_ylabel('Average Score',fontsize=15)
        c+=1
    plt.legend(title="Hierarchy of Needs",
              loc="right",
              bbox_to_anchor=(1, 0, 0.42, 1),fontsize=10)
    plt.subplots_adjust(wspace =0.1, hspace =0.25)#调整子图间距
    plt.show()

df_score = pd.read_csv('item.csv',index_col=[0],parse_dates = True)
#Draw_Score(3,2,0,6,df_score)
# Draw_Score(3,2,5,11,df_score)
#Draw_Score(3,2,11,17,df_score)
#Draw_Score(3,2,17,22,df_score)
#Draw_Score(3,2,22,28,df_score)

def Draw_Kmeans(column,row,first,last,dataframe):
    dataframe.sample(frac=1).reset_index(drop=True)
    x = dataframe.iloc[:,1:].values
    cmap = plt.get_cmap("tab20")
    filled_markers = ['o', 'v','>', '8', 's', '*', 'H', 'D', 'P', 'X','d','h']
    fig = plt.figure(figsize=(12,6))


    for n_c in range(first,last+1,1):
        pca = decomposition.PCA(n_components=3).fit(x)
        reduced_X = pca.transform(x)
        estimator = KMeans(n_clusters=n_c, init='k-means++')#构造聚类器
        estimator.fit(reduced_X)#聚类
        label_pred = estimator.labels_
        ax = fig.add_subplot(1,2,n_c-first+1, projection='3d')

        if n_c%2 == 0:turn = 0.9
        else:turn = -0.25

        for i in range(28):
            colors = cmap(np.arange(12)/10)
            markers = filled_markers[label_pred[i]]
            ax.scatter(reduced_X[i][0], reduced_X[i][1], reduced_X[i][2],c = colors[label_pred[i]],marker=markers,s=80,label= dataframe.iloc[i].values[0],edgecolor='k')
            ax.set_title('N_Clusters=%s'%(n_c),fontsize=20,fontweight='bold',y=-0.1)

        plt.legend(loc='center left',fontsize=7,
        bbox_to_anchor=(turn, 0, 0, 1))
    plt.subplots_adjust(wspace =-0.1, hspace =0)#调整子图间距
    plt.show()

# Draw_Kmeans(2,1,5,6,df_hons)
# Draw_Kmeans(2,1,7,8,df_hons)
# Draw_Kmeans(2,1,9,10,df_hons)
#Draw_Kmeans(2,1,11,12,df_hons)

# plt.figure(figsize=(9,4))
# SSE = []
x = df_hons.iloc[:,1:].values
# for k in range(1,15):
#     estimator = KMeans(n_clusters=k)  # 构造聚类器
#     estimator.fit(x)
#     SSE.append(estimator.inertia_)
# X = range(1,15)
# plt.subplot(1, 2, 1)#PCA ROC
# plt.xlabel('k',fontsize=10,fontweight='bold')
# plt.ylabel('sum of the squared errors',fontsize=10,fontweight='bold')
# plt.xticks(np.linspace(1,15,15))
# plt.title('Error sum of squared with K clusters')
# plt.plot(X,SSE,'o-')
# #plt.show()
#
# plt.subplot(1, 2, 2)#PCA ROC
# Scores = []  # 存放轮廓系数
# for k in range(2,16):
#     estimator = KMeans(n_clusters=k)  # 构造聚类器
#     estimator.fit(x)
#     Scores.append(silhouette_score(x,estimator.labels_,metric='euclidean'))
# X = range(2,16)
# plt.subplot(1, 2, 2)#PCA ROC
# plt.xlabel('k',fontsize=10,fontweight='bold')
# plt.xticks(np.linspace(1,15,15))
# plt.ylabel('Silhouette Coefficient',fontsize=10,fontweight='bold')
# plt.title('Silhouette Coefficient with K clusters')
# plt.plot(X,Scores,'o-')
# plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from matplotlib import pyplot as plt

#linked = linkage(x, 'single')
plt.figure(figsize=(40, 10))
labelList = df_hons.iloc[:,0].values
methodlist = ['single','complete','average','ward']
for i in range(4):
    linked = linkage(x, method=methodlist[i])
    #
    plt.subplot(4, 1, i+1)
    plt.title("Game Classification Dendograms with %s assigns"%(methodlist[i]),fontweight='bold')
    dend = shc.dendrogram(shc.linkage(x, method=methodlist[i]),labels=labelList)
    plt.xticks(fontsize=15)
plt.subplots_adjust(hspace =1.8)#调整子图间距
    #plt.xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],['0',r'$\frac{\pi}{2}$',r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$'], rotation=90)
plt.show()

# for i in range(4):
#         ARIs=[]
#         for _ in range(1,13,1):
#             linked = linkage(x, method=methodlist[i])
#             clst=cluster.AgglomerativeClustering(n_clusters=_,linkage=linked)
#             predicted_labels=clst.fit_predict(x)
#             ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
#         ax.plot(nums,ARIs,marker=markers[i],label="linkage:%s"%linkage)

# from sklearn.cluster import DBSCAN
# import numpy as np
#
# pca = decomposition.PCA(n_components=2).fit(x)
# reduced_X = pca.transform(x)
# clustering = DBSCAN(eps=0.5, min_samples=2).fit(x)
# labelList = df_hons.iloc[:,0].values
# cl = clustering.fit_predict(x)#聚类
# print(cl)
# cmap = plt.get_cmap("tab20")
# filled_markers = ['o', 'v','>', '8', 's', '*', 'H', 'D', 'P', 'X','d','h','<','1','2','3','4','+','x','p']
# colors = cmap(np.arange(len(cl))/10)
# plt.subplot(1, 2, 1)
# for i in range(28):
#     print(cl[i])
#     plt.scatter(reduced_X[i][0], reduced_X[i][1], c=colors[cl[i]],marker = filled_markers[cl[i]],edgecolor='k',label=labelList[i],s=50)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(loc='center left',fontsize=6,bbox_to_anchor=(1, 0, 0, 1))
#
#
# clustering = DBSCAN(eps=0.55, min_samples=2).fit(x)
# labelList = df_hons.iloc[:,0].values
# cl = clustering.fit_predict(x)#聚类
# print(cl)
# cmap = plt.get_cmap("tab20")
# filled_markers = ['o', 'v','>', '8', 's', '*', 'H', 'D', 'P', 'X','d','h','<','1','2','3','4','+','x','p']
# colors = cmap(np.arange(len(cl))/10)
# plt.subplot(1, 2, 2)
# for i in range(28):
#     print(cl[i])
#     plt.scatter(reduced_X[i][0], reduced_X[i][1], c=colors[cl[i]],marker = filled_markers[cl[i]],edgecolor='k',label=labelList[i],s=50)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(loc='center left',fontsize=6,bbox_to_anchor=(1, 0, 0, 1))
# plt.show()
