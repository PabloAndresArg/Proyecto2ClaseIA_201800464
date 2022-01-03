#201800464 PABLO ANDRES ARGUETA HERNANDEZ
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[8,2],[9,7],[2,12], [9,1], [10,7] , [3,14] ,[8,1] ,[1,13]])

#                                     primera  iteracion
# CENTROIDE = np.array([[10,7],])
# kmeans = KMeans(n_clusters=1, init=CENTROIDE, n_init=1)
# kmeans.fit(X)
# plt.show()

#                                     segunta iteracion 
# CENTROIDE = np.array([[10,7],[6.25,7.125]])
# kmeans = KMeans(n_clusters=2, init=CENTROIDE , n_init=1)
# kmeans.fit(X)
# plt.show()

#                                       tercerda iteracion
# CENTROIDE = np.array([[10,7],[8.8,3.6],[2,13]])
# kmeans = KMeans(n_clusters=3, init=CENTROIDE , n_init=1)
# kmeans.fit(X)
# plt.show()




Kmeans = KMeans(n_clusters=3)
Kmeans.fit(X)
print(Kmeans.cluster_centers_)
plt.scatter(X[:,0],X[:,1] , c =Kmeans.labels_ , cmap='rainbow')
# plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],color="blue")


# predicciones:
# predicts = kmeans.predict(X)
# print("***********************")
# print("predicciones: ")
# print(predicts)
# print("***********************")
plt.show()