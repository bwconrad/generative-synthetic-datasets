import matplotlib.pyplot as plt
import numpy as np
from synthetic_dataset import GridGaussianDataset
from synthetic_dataset import CircularGaussianDataSet
from synthetic_dataset import ArchimedeanSpiralDataSet
from synthetic_dataset import * 

#data = GridGaussianDataset(rows=2, cols=2, sample_weights=[1,1,2,1], samples = 5)
#data = CircularGaussianDataSet(modes=7, sample_weights=[1,2,1,1,1,1,1], samples = 8)
data = ArchimedeanSpiralDataSet(revolutions=2, scale=0.5)

x, y = zip(*data)
plt.figure(figsize=(8,8))
#plt.scatter(x,y)
plt.scatter(x, y, s=3, color='black', alpha=0.1)
plt.show()


'''
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8).fit(data)
counts = np.bincount(kmeans.labels_)

print(counts)
plt.bar([i+1 for i in range(len(counts))],counts)
plt.show()
'''