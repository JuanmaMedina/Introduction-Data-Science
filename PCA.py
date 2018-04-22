from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

                                                ## PCA ##
# EXERCISE 1: Performing PCA

# Load the "murdered" dataset
murdered = np.loadtxt('/home/juanma/IDS/murderdata2d.txt')
murdered = np.ndarray.transpose(murdered)

# Data means and centering
xm = np.mean(murdered[0, :])
ym = np.mean(murdered[1, :])
x = murdered[0, :] - xm
y = murdered[1, :] - ym

# a) PCA function implementation

def pca(datamatrix):
    '''Takes a datamatrix and returns its unit vectors spanning their PCs and the variance captured by each one'''

    # Covariance matrix
    Sigma = np.cov(datamatrix)

    # Eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eig(Sigma)

    return evecs, evals

# b) PCA and scatterplot with mean and PC (with length scaled by the sd of the data projected onto that vector)

# print pca(murdered)
# Eigenvectors:  array([[-0.99442433, -0.1054526 ],       Variance captured by each PC:
#                       [ 0.1054526 , -0.99442433]])               array([0.36319983,  98.73614228])

# Covariance matrix Sigma from the data
Sigma = np.cov(murdered)

# Eigenvalue decomposition of Sigma
evals, evecs = np.linalg.eig(Sigma)

# Standard deviations of the eigenvalues
s0 = np.sqrt(evals[0])
s1 = np.sqrt(evals[1])

# Plot the projected variance on each PC with their length scaled by the standard deviation
plt.scatter(x, y)
plt.axis('equal')
plt.plot([0, s0*evecs[0,0]], [0, s0*evecs[1,0]], 'r')
plt.plot([0, s1*evecs[0,1]], [0, s1*evecs[1,1]], 'r')
plt.title('PCA of "murdered" dataset')

plt.show()



# c) PCA, plot of variance VS PC index and plot of cumulative variance vs PC index on the "pesticide" dataset

# Load the "pesticide" dataset
pesticide = np.loadtxt('/home/juanma/IDS/IDSWeedCropTrain.csv', delimiter=',')
pesticide = pesticide[:, :-1]
pesticide = np.ndarray.transpose(pesticide)

# print pca(pesticide) # --> Only reporting the variance captured by each PC (eigenvalues, y1), not the eigenvectors
y1 = [3.40409799e+06,  1.37998456e+06,  2.42564152e+05,  1.18244260e+05,  4.70092000e+04,
      1.54582538e+04,  5.56336848e+03,  3.18377319e+03,  1.23984337e+03,  1.00877066e+03,
      1.13586752e-03,  3.40310304e+01,  2.63012185e+01]

x1 = range(len(y1))

plt.plot(x1, y1)
plt.xlabel('PC index')
plt.ylabel('Variance')
plt.title('Variance of "pesticide" dataset VS PC index')

plt.show()

# Covariance matrix of pesticide dataset and its eigenvalue decomposition
Sigma2 = np.cov(pesticide)
evals2, evecs2 = np.linalg.eig(Sigma2)

# Compute and plot the cumulative variance in percent, checking how much of the variance can be described by each PC
y2 = np.cumsum(evals2 / np.sum(evals2))

# print y2
# [0.65232418,  0.91676936,  0.96325171,  0.98591075,  0.99491908,
#  0.99788133,  0.99894743,  0.99955754,  0.99979513,  0.99998844,
#  0.99998844,  0.99999496,  1.00000000]

x2 = range(len(y2))

plt.plot(x2, y2)
plt.xlabel('PC index')
plt.ylabel('Cumulative variance')
plt.title('Cumulative variance of "pesticide" dataset VS PC index')

plt.show()

## CONCLUSION: To capture the 90% and 95% of the variance, 2 and 3 PCs/dimensions respectively are required. ##


# EXERCISE 2: Visualization in 2D

def mds(data):
    '''Takes a datamatrix and returns a N x 2 matrix with the coord. of the N datapoints projected
                                    onto the top of the 2 first PCs'''
    # Data number of columns
    cols = np.shape(data)[1]

    # Initialize empty matrix with number of columns as rows and 0 columns
    empty_matrix = np.empty((cols, 0), int)

    # First two PCs of the matrix (by calling pca function defined in 1-a): columns in the eigenvectors matrix
    PC1 = pca(data)[0][:,0]
    PC2 = pca(data)[0][:,1]

    # Compute the projected datapoints onto the top of the 2 first PCs and reshape as columns with 1000 new datapoints
    col_1 = np.dot(PC1, data).reshape(1000, 1)
    col_2 = np.dot(PC2, data).reshape(1000, 1)

    # Append the projected datapoints onto the top of the 2 first PCs to the empty matrix
    final_matrix = np.concatenate((empty_matrix, col_1, col_2), axis=1)

    return final_matrix

x3 = mds(pesticide)[:, 0]
y3 = mds(pesticide)[:, 1]

plt.scatter(x3, y3)
plt.axis('equal')
plt.title('Projected "pesticide" dataset onto the top of the 2 first PCs')

plt.show()











