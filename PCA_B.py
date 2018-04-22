from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


                                                ## DATA EXPLORATION WITH PCA ##
# EXERCISE 1: Plotting cell shapes

# Load the "diatoms" dataset
diatoms = np.loadtxt('/home/juanma/IDS/diatoms.txt')

# Separate out the x (even columns) and y (odd columns) coordinates
diatoms_xcoords = diatoms[:, ::2]
diatoms_ycoords = diatoms[:, 1::2]

# Plot of the first cell with its landmark points connected
x0 = diatoms_xcoords[0]
y0 = diatoms_ycoords[0]

plt.plot(x0, y0, '-o')
plt.axis('equal')
plt.title('Plot of one single diatom')

plt.show()

# Plot of all the cells superimposed with their landmark points connected
for i in range(len(diatoms_xcoords)):
    plt.plot(diatoms_xcoords[i], diatoms_ycoords[i], '-o')
    plt.axis('equal')
    plt.title('Plot of all superimposed diatoms')

plt.show()


# EXERCISE 2: Plot the five cells corresponding to the first three PCs in a single plot each one

# Compute the mean diatom
m = np.mean(np.ndarray.transpose(diatoms), 1)

# Perform PCA on the dataset
Sigma = np.cov(np.ndarray.transpose(diatoms))
evals, evecs = np.linalg.eig(Sigma)

# Compute the cumulative variance in percent, checking how much of the variance is described by the first 3 PCs
c_var = np.cumsum(evals/np.sum(evals))

print c_var[0:3] # [0.77187165  0.92769958  0.95211976]

# Standard deviations of the first 3 eigenvalues
std1 = np.sqrt(evals[0])
std2 = np.sqrt(evals[1])
std3 = np.sqrt(evals[2])

# First 3 eigenvectors
e1 = evecs[:, 0]
e2 = evecs[:, 1]
e3 = evecs[:, 2]

# Delimit the range of the diatoms shape with 180 data points
x = range(0, 180, 2)
y = range(1, 180, 2)

# Empty points to be filled
diatom_along_pc = np.zeros((6,180))

# Set the mean and standard deviation of the first PC
for i in range(6):
    diatom_along_pc[i,:] = m + (i-3) * std1 * e1

# Plot the shapes along the first PC
for i in range(1,6):
    plt.plot(diatom_along_pc[i][x], diatom_along_pc[i][y])
    plt.axis('equal')

plt.title('Plot of the data mean and its variance along the first PC')
plt.show()


# Set the mean and standard deviation of the second PC
for i in range(6):
    diatom_along_pc[i,:] = m + (i-3) * std2 * e2

# Plot the shapes along the second PC
for i in range(1,6):
    plt.plot(diatom_along_pc[i][x], diatom_along_pc[i][y])
    plt.axis('equal')

plt.title('Plot of the data mean and its variance along the second PC')
plt.show()


# Set the mean and standard deviation of the third PC
for i in range(6):
    diatom_along_pc[i,:] = m + (i-3) * std3 * e3

# Plot the shapes along the third PC
for i in range(1,6):
    plt.plot(diatom_along_pc[i][x], diatom_along_pc[i][y])
    plt.axis('equal')

plt.title('Plot of the data mean and its variance along the third PC')
plt.show()


# EXERCISE 3: Perform PCA on the "toydata" dataset and visualize the projection onto the first 2 PCs

def pca(datamatrix):
    '''Takes a datamatrix and returns its unit vectors spanning their PCs and the variance captured by each one'''

    # Covariance matrix
    Sigma = np.cov(datamatrix)

    # Eigenvalues and eigenvectors of the covariance matrix
    evals, evecs = np.linalg.eig(Sigma)

    return evecs, evals


def mds(data):
    '''Takes a datamatrix and returns a N x 2 matrix with the coord. of the N datapoints projected
                                    onto the top of the 2 first PCs'''
    # Data number of columns
    cols = np.shape(data)[1]

    # Initialize empty matrix with number of columns as rows and 0 columns
    empty_matrix = np.empty((cols, 0), int)

    # First two PCs of the matrix (by calling pca function defined above)
    PC1 = pca(data)[0][:,0]
    PC2 = pca(data)[0][:,1]

    # Compute the projected datapoints onto the top of the 2 first PCs and reshape as columns with 1000 new datapoints
    col_1 = np.dot(PC1, data).reshape(1000, 1)
    col_2 = np.dot(PC2, data).reshape(1000, 1)

    # THIS PART OF THE FUNCTION IS TO DEAL WITH THE TOYDATASET #
    # JUST SUBSTITUTE THE COL_1 AND COL_2 PARAMETERS INSTEAD OF THE PREVIOUS ONES TO RESHAPE THE MATRIX #

    # Compute the projected datapoints onto the top of the 2 first PCs and reshape as columns with 102 new datapoints
    # col_1 = np.dot(PC1, data).reshape(102, 1)
    # col_2 = np.dot(PC2, data).reshape(102, 1)

    # THIS PART OF THE FUNCTION IS TO DEAL WITH THE MODIFIED TOYDATASET (WITHOUT THE LAST 2 COLUMNS) #
    # JUST SUBSTITUTE THE COL_1 AND COL_2 PARAMETERS INSTEAD OF THE PREVIOUS ONES TO RESHAPE THE MATRIX #

    # Compute the projected datapoints onto the top of the 2 first PCs and reshape as columns with 100 new datapoints
    # col_1 = np.dot(PC1, data).reshape(100, 1)
    # col_2 = np.dot(PC2, data).reshape(100, 1)

    # Append the projected datapoints onto the top of the 2 first PCs to the empty matrix
    final_matrix = np.concatenate((empty_matrix, col_1, col_2), axis=1)

    return final_matrix


# Load the "toydata" dataset
toydata = np.loadtxt('/home/juanma/IDS/pca_toydata.txt')
toydata = np.ndarray.transpose(toydata)

## CHANGE NECESSARY IN THE MDS FUNCTION TO READJUST THE MATRIX TO THE NEW PURPOSES (102 COLUMNS)

"""x1 = mds(toydata)[:, 0]
y1 = mds(toydata)[:, 1]

plt.scatter(x1, y1)
plt.axis('equal')
plt.title('Projected "toydata" dataset onto the top of the 2 first PCs')

plt.show()"""


# Repeat the procedure leaving out the 2 last datapoints

# Modify the "toydata" dataset, removing the last 2 columns
toydata_modi = toydata[:, :-2]

## CHANGE NECESSARY IN THE MDS FUNCTION TO READJUST THE MATRIX TO THE NEW PURPOSES (100 COLUMNS)

"""x2 = mds(toydata_modi)[:, 0]
y2 = mds(toydata_modi)[:, 1]

plt.scatter(x2, y2)
plt.axis('equal')
plt.title('Projected "toydata" modified dataset onto the top of the 2 first PCs')

plt.show()"""


                                            ## CLUSTERING II ##

# Project the "pesticide" dataset onto the top of the first two PCs, color the data points (crop and weed separately),
# and project the centroids found in Exercise 3.3 onto the top of these PCs

# Load the "pesticide" dataset
pesticide = np.loadtxt('/home/juanma/IDS/IDSWeedCropTrain.csv', delimiter=',')

# Extract the labels (last column)
labels = pesticide[:, -1]

# Transpose the matrix so it fits the mds function
pesticide = np.transpose(pesticide)

x = mds(pesticide)[:, 0]
y = mds(pesticide)[:, 1]

# Iterate over the labels array, find their corresponding x,y values and apply a differential color
for i in range(len(labels)):
    if labels[i] == 0:
        plt.scatter(x[i], y[i], c='cyan')
    else:
        plt.scatter(x[i], y[i], c='green')

# Centroid coordinates found in Exercise 3.3 with the built-in function
centroid_1 = np.array([5.69426752e+00, 4.93800425e+01, 7.91594480e+02, 3.84771338e+03,
                       3.38588535e+03, 1.35988535e+03, 2.93734607e+02, 1.31609342e+02,
                       7.07282378e+01, 3.96433121e+01, 1.94437367e+01, 4.23566879e+00,
                       4.41613588e-01])

centroid_2 = np.array([2.19092628e+00, 1.37315690e+01, 1.70943289e+02, 1.39436484e+03,
                       3.18853497e+03, 2.62461815e+03, 1.00372023e+03, 6.32814745e+02,
                       4.95829868e+02, 2.95400756e+02, 1.45809074e+02, 2.91984877e+01,
                       2.83742911e+00])

# Mean of each coordinate of the "pesticide" dataset
mean_pesticide = np.mean(pesticide, axis=1)[:-1]

# Centroids coordinates centering
centroid_1 = centroid_1 - mean_pesticide
centroid_2 = centroid_2 - mean_pesticide

# Matrix of centroid coordinates
centroids_matrix = np.column_stack((centroid_1, centroid_2))

# 2 first PCs of the pesticide dataset
PC_cen1 = pca(pesticide)[0][:,0]
PC_cen2 = pca(pesticide)[0][:,1]

# Projection of the centroids onto the top of the PCs (excluding the labels of the pesticide dataset)
centroid1_red = np.dot(PC_cen1[:-1], centroids_matrix)
centroid2_red = np.dot(PC_cen2[:-1], centroids_matrix)

# Plotting of the centroids together with the datapoints separated by class
plt.scatter(centroid1_red[0], centroid1_red[1], marker='x', c='red')
plt.scatter(centroid2_red[0], centroid2_red[1], marker='x', c='red')

plt.axis('equal')
plt.title('Projected "pesticide" dataset onto the top of the 2 first PCs with the centroids coordinates')

plt.show()
