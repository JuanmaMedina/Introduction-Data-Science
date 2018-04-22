from __future__ import division
import numpy as np
from sklearn.cluster import KMeans
                                                ## Clustering ##

# Load the input vectors (S)
data = np.loadtxt('/home/juanma/IDS/IDSWeedCropTrain.csv', delimiter=',')

# CUSTOM IMPLEMENTATION OF THE K-MEANS ALGORITHM
S = data[:, :-1]

# Set the two clusters
S1 = []
S2 = []

# Initialize cluster centroids with the two first datapoints
k1 = S[0,]
k2 = S[1,]

# DATA ASSIGNMENT
# If a data is nearer from centroid k1 than from centroid k2, assign it to S1 cluster, and vice versa
for data in S:
    if (np.linalg.norm(data - k1)**2) < (np.linalg.norm(data - k2)**2):
        S1.append(data)
    else:
        S2.append(data)

# CENTROID RELOCATION
k1_reloc = (1 / len(S1)) * sum(S1)
k2_reloc = (1 / len(S2)) * sum(S2)

print k1_reloc, k2_reloc

# [1.92708333e-01   3.41145833e+00   1.40458333e+02   2.59171875e+03
#  4.77511979e+03   2.01004688e+03   3.26015625e+02   9.77552083e+01
#  3.53177083e+01   1.38333333e+01   5.22916667e+00   8.17708333e-01
#  8.33333333e-02]

# [4.70792079e+00   3.69641089e+01   5.39977723e+02   2.53995297e+03
#  2.92656436e+03   2.03341584e+03   7.50893564e+02   4.67794554e+02
#  3.57457921e+02   2.13221535e+02   1.05553218e+02   2.13910891e+01
#  2.09529703e+00]


# IMPLEMENTATION OF THE K-MEANS ALGORITHM WITH BUILT-IN FUNCTION FROM SCIKIT

# Initialize cluster centroids with the two first datapoints
startingPoint = np.vstack((S[0,], S[1,]))

# Set-up k-means clustering algorithm with built-in function (2 clusters, 1 set of initial cluster centers)
kmeans = KMeans(n_clusters=2, init=startingPoint, n_init=1, algorithm='full').fit(S)

print kmeans.cluster_centers_

# [5.69426752e+00   4.93800425e+01   7.91594480e+02   3.84771338e+03
#  3.38588535e+03   1.35988535e+03   2.93734607e+02   1.31609342e+02
#  7.07282378e+01   3.96433121e+01   1.94437367e+01   4.23566879e+00
#  4.41613588e-01]

# [2.19092628e+00   1.37315690e+01   1.70943289e+02   1.39436484e+03
#  3.18853497e+03   2.62461815e+03   1.00372023e+03   6.32814745e+02
#  4.95829868e+02   2.95400756e+02   1.45809074e+02   2.91984877e+01
#  2.83742911e+00]





