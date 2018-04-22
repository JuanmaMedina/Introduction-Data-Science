from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


## LINEAR REGRESSION

# Exercise 1: Predict the quality score of the wine from its physicochemical properties using LR

def multivarlinreg(X, y):
    '''This function takes an (N x D)-dimensional data matrix of independent variables (X) and a vector of ground truth
    dependant values (y) as numpy arrays and returns the regression coefficients (D+1)-dimensional vector (w)'''

    # Append a vector of ones as the first column of X
    onevec = np.ones((len(y), 1))
    X = np.concatenate((onevec, X), axis=1)

    # Compute the regression coefficients implementing the analytic solution deduced from the slides
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)

    return w


# Load the wine dataset
data_train = np.loadtxt('/home/juanma/IDS/redwine_training.txt')
data_test = np.loadtxt('/home/juanma/IDS/redwine_testing.txt')

# Independant variables (X; physicochemical properties)
feature1 = data_train[:, [0]]
allfeatures = data_train[:, :-1]

# Predicted dependant variable (y; quality score)
quality = data_train[:, -1]

print multivarlinreg(feature1, quality)
print multivarlinreg(allfeatures, quality)


# Exercise 2: Evaluating LR

def rmse(f, y):
    '''This function takes an N-dimensional vector of predicted dependant values (f) and an N-dimensional vector of
     ground truth dependant values (y) as numpy arrays and returns the RMSE as a 1 x 1 numpy array'''

    RMSE = np.sqrt((sum(f - y)**2) / len(f))

    return RMSE


# First feature independent values
one_vec = np.ones((len(feature1), 1))
x1 = np.concatenate((one_vec, feature1), axis=1)

# Regression coefficients for the first feature
w1 = multivarlinreg(feature1, quality)

# Predicted dependant values for the first feature (f = x * w)
f1 = np.dot(x1, w1)

print rmse(f1, quality)


# All features independent values
allx = np.concatenate((one_vec, allfeatures), axis=1)

# Regression coefficients for all features
allw = multivarlinreg(allfeatures, quality)

# Predicted dependant values for all features (f = x * w)
allf = np.dot(allx, allw)

print rmse(allf, quality)


# Exercise 4: Apply a RF classifier to the data and determine the classification accuracy

# CSV files with the train and test datasets
dataTrain = np.loadtxt('/home/juanma/IDS/IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('/home/juanma/IDS/IDSWeedCropTest.csv', delimiter=',')

# 4 arrays with input variables and labels of train and test datasets respectively
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, :-1]
YTest = dataTest[:, -1]

# Set-up RF classifier with built-in function and 50 trees
clf = RandomForestClassifier(n_estimators=50)

# Train the classifier using XTrain as training data and YTrain as labels
clf.fit(XTrain, YTrain)

# Determine the model accuracy on the given test dataset by comparing the real labels VS model predicted labels
accTest = accuracy_score(YTest, clf.predict(XTest))

print accTest


# Exercise 5: Apply GD to find the minimum of the given function using these the learning rates indicated below

def f_derivative(x):
    '''This function computes the gradient of the objective function'''
    return 20 * x - (math.exp(-x / 2) / 2)

w = 1                      # Initialization
num_iter = 0               # Number of iterations counter
learningrate = 0.01        # Size of the steps taken  --> test with (0.1, 0.01, 0.001, 0.0001)
tolerance = 0.001          # Convergence 1st criteria = updated difference in the objective function < tolerance
max_iter = 10000           # Convergence 2nd criteria = maximum number of iterations allowed (for subexercise c)
convergence = 0            # Binary variable to check convergence (reached if convergence = 1)

# Record the objective function values and the gradient descent steps at each iteration
values = []
gradient_steps = []

while convergence == 0:
    # Gradient computation (by calling f_derivative defined above)
    gradient = f_derivative(w)

    # Step size applied to gradient
    move = gradient * learningrate

    # Temporary local minimum, found by moving into the direction of the steepest descent relative to the gradient
    cur_w = w - move

    # Compute and record objective function value at current iteration, as well as gradient descent steps
    values.append((3 - cur_w) ** 2)
    gradient_steps.append(cur_w)

    # Add 1 iteration to the counter
    num_iter = num_iter + 1

    # Compute the difference in the objective function
    diff = abs(cur_w - w)

    # Apply the two criteria defined above to reach the convergence
    if diff < tolerance:
        convergence = 1
    elif num_iter >= 3:
        convergence = 1

    # Update of the local minimum
    w = cur_w

print "Local minimum reached occurs at: %s" % round(w, 2)
print "Number of iterations: %s" % num_iter

plt.plot(gradient_steps, '-g', label='Gradient descent steps')
plt.plot(values, '-o', label='Objective function values')
plt.xlabel('Number of iterations')
plt.ylabel('Value of f')
plt.title('Gradient descent steps (learning rate = 0.1, iterations = 3)')
plt.legend(loc='center right')

plt.show()


# Exercise 6: Implement, run and test logistic regression on the two Iris datasets

# Load the Iris dataset
iris1_train = np.loadtxt('/home/juanma/IDS/Iris2D1_train.txt')
iris1_test = np.loadtxt('/home/juanma/IDS/Iris2D1_test.txt')

iris2_train = np.loadtxt('/home/juanma/IDS/Iris2D2_train.txt')
iris2_test = np.loadtxt('/home/juanma/IDS/Iris2D2_test.txt')

# Load the 4 datasets
iris1_train_data = iris1_train[:,[0,1]]
iris1_train_labels = iris1_train[:,2]

iris2_train_data = iris2_train[:,[0,1]]
iris2_train_labels = iris2_train[:,2]

iris1_test_data = iris2_test[:,[0,1]]
iris1_test_labels = iris2_test[:,2]

iris2_test_data = iris2_test[:,[0,1]]
iris2_test_labels = iris2_test[:,2]

# Plot the 4 datasets, with the following labels: red = label 0, blue = label 1
cl0_iris1 = np.where(iris1_train_labels==0)
cl1_iris1 = np.where(iris1_train_labels==1)
data_matrix0_iris1 = iris1_train_data[cl0_iris1]
data_matrix1_iris1 = iris1_train_data[cl1_iris1]

plt.scatter(data_matrix0_iris1[:,0], data_matrix0_iris1[:,1], c='r')
plt.scatter(data_matrix1_iris1[:,0], data_matrix1_iris1[:,1], c='b')
plt.title('Iris2D1 train dataset')

plt.show()

cl0_iris2 = np.where(iris2_train_labels==0)
cl1_iris2 = np.where(iris2_train_labels==1)
data_matrix0_iris2 = iris2_train_data[cl0_iris2]
data_matrix1_iris2 = iris2_train_data[cl1_iris2]

plt.scatter(data_matrix0_iris2[:,0], data_matrix0_iris2[:,1], c='r')
plt.scatter(data_matrix1_iris2[:,0], data_matrix1_iris2[:,1], c='b')
plt.title('Iris2D2 train dataset')

plt.show()

cl0_iris1_test = np.where(iris1_test_labels==0)
cl1_iris1_test = np.where(iris1_test_labels==1)
data_matrix0_iris1_test = iris1_test_data[cl0_iris1_test]
data_matrix1_iris1_test = iris1_train_data[cl1_iris1_test]

plt.scatter(data_matrix0_iris1_test[:,0], data_matrix0_iris1_test[:,1], c='r')
plt.scatter(data_matrix1_iris1_test[:,0], data_matrix1_iris1_test[:,1], c='b')
plt.title('Iris2D1 test dataset')

plt.show()

cl0_iris2_test = np.where(iris2_test_labels==0)
cl1_iris2_test = np.where(iris2_test_labels==1)
data_matrix0_iris2_test = iris2_test_data[cl0_iris2_test]
data_matrix1_iris2_test = iris2_train_data[cl1_iris2_test]

plt.scatter(data_matrix0_iris2_test[:,0], data_matrix0_iris2_test[:,1], c='r')
plt.scatter(data_matrix1_iris2_test[:,0], data_matrix1_iris2_test[:,1], c='b')
plt.title('Iris2D2 test dataset')

plt.show()

# Defined functions used in the implementation of the logistic regression solved by gradient descent
# All of them have been taken from the templates provided in Jupyter notebook number 12

def logistic_function(input):
    '''This function takes some data as input, introduces it in the logistic function and returns the result'''

    # Definition of logistic function
    result = 1 / (1 + np.exp(-input))

    return result

def logistic_LLH(X, y, w):
    '''This function takes an N x (d+1) dimensions data matrix (X), an N x 1 dimensions vector (y) and a (d+1) x 1
    dimensions weight vector (w), introduces them in the E(in) objective function and returns the logistic LLH'''

    # Dimensions of the data matrix
    N, d = X.shape

    # E(in) objective function initialization
    E = 0

    # Apply the objective function, as defined in the slides, to every trio (x1n, x2n, yn) of the input data matrix,
    # that is, for every row containing the information about the two features and the label (in this specific case)
    for n in range(N):
        E += (1/N) * np.log(1 + np.exp(-y[n] * np.dot(w, X[n,:])))

    return E

def logistic_gradient(X, y, w):
    '''This function takes an N x (d+1) dimensions data matrix (X), an N x 1 dimensions vector (y) and a (d+1) x 1
    weight vector (w) and returns its logistic gradient, applying the 1st derivative of the E(in) objective function'''
    N = X.shape[0]

    # Gradient initialization
    grad = 0

    # Apply the gradient (1st derivative of the E(in) function), as defined in the slides, to every trio (x1n, x2n, yn)
    # of the input data matrix (in this specific case)
    for n in range(N):
        grad += ((-1/N) * y[n] * X[n,:]) * logistic_function(-y[n] * np.dot(w, X[n,:]))

    return grad


# GRADIENT DESCENT IMPLEMENTATION
def logistic_regression(X, y, max_iter, grad_thr):
    '''This function takes an N x (d+1) dimensions data matrix (X) and N x 1 dimensions vector (y), performs
    logistic gradient descent until a maximum number of iterations or a gradient threshold are reached and returns
            the best weights found and the objective function values obtained with each weight'''

    # Dimensions of the input data matrix with a column of 1 appended (to reach the first weight w0)
    N, d = X.shape
    onevec = np.ones((N, 1))
    X = np.concatenate((onevec, X), axis=1)

    # Dimensions of the input label vector, with labels switched to -1 and 1 instead of 0 and 1
    y = np.array((y - 0.5)*2)

    # Initialize learning rate for gradient descent
    learningrate = 0.01

    # Initialize weights at random in step 0
    w = 0.1 * np.random.randn(d + 1)

    # Compute value of logistic log likelihood
    value = logistic_LLH(X, y, w)

    # Number of iterations initialization and binary variable to check convergence (reached if convergence = 1)
    num_iter = 0
    convergence = 0

    #Record the objective function values
    E = []

    while convergence == 0:
        num_iter = num_iter + 1

        # Compute gradient with current w
        grad = logistic_gradient(X, y, w)

        # Set direction to move
        v = -grad

        # Update weights
        w_new = w + learningrate * v

        # Compute value of logistic log likelihood with new w
        cur_value = logistic_LLH(X, y, w_new)

        # Check improvement by comparing LLH with actual and previous w. If the in-sample error with actual w is lower
        # than with the previous w, update w and LLh, record the objective function value and take "longer" steps. Else,
        # it could mean we have gone too fast and missed better points, so change to "shorter" steps
        if cur_value < value:
            w = w_new
            value = cur_value
            E.append(value)
            learningrate *= 1.1
        else:
            learningrate *= 0.9

        # Normalize gradient
        g_norm = np.linalg.norm(grad)

        # Determine convergence by cheking if gradient is below threshold and/or max_iter have been reached
        if g_norm < grad_thr:
            convergence = 1
        elif num_iter > max_iter:
            convergence = 1

    return w

# The three parameters of the affine linear model applied to the train and test datasets of Iris2D1 and Iris 2D2
w_iris1_train = logistic_regression(iris1_train_data, iris1_train_labels, 1000, 0.0000)
w_iris1_test = logistic_regression(iris1_test_data, iris1_test_labels, 1000, 0.0000)
w_iris2_train = logistic_regression(iris2_train_data, iris2_train_labels, 1000, 0.0000)
w_iris2_test = logistic_regression(iris2_test_data, iris2_test_labels, 1000, 0.0000)

print "The three parameters of the affine linear model applied to the train Iris2D1 dataset: %s" % w_iris1_train
print "The three parameters of the affine linear model applied to the test Iris2D1 dataset: %s" % w_iris1_test
print "The three parameters of the affine linear model applied to the train Iris2D2 dataset: %s" % w_iris2_train
print "The three parameters of the affine linear model applied to the test Iris2D2 dataset: %s" % w_iris2_test


N_train1 = iris1_train_data.shape[0]
N_test1 = iris1_test_data.shape[0]
N_train2 = iris2_train_data.shape[0]
N_test2 = iris2_test_data.shape[0]


def logistic_prediction(X, w):
    ''' This function takes an N x (d+1) dimensions data matrix (X) and a (d+1) x 1 weight vector (w) and returns a
            matrix with the logistic function results and the predicted binary classes of the input data'''

    # Dimensions of the input data matrix with a column of 1 appended (to reach the first weight w0)
    N = X.shape[0]
    onevec = np.ones((N, 1))
    X = np.concatenate((onevec, X), axis=1)

    # Re-defined number of rows as empty vector
    N = X.shape[0]
    P = np.zeros(N)

    # Fill the empty vector with the logistic function results
    for n in range(N):
        arg = np.exp(np.dot(w, X[n, :]))
        prob_i = arg / (1 + arg)
        P[n] = prob_i

    # Empty vector equivalent to P with the 0 and 1 class labels
    y = np.round(P)

    # Labels switched to -1 and 1 instead of 0 and 1
    y = (y - 0.5) * 2

    return P, y

# Compute the predicted binary classes of both input training data with an input data matrix and a weight vector
P_iris1_train, y_iris1_train = logistic_prediction(iris1_train_data, w_iris1_train)
P_iris1_test, y_iris1_test = logistic_prediction(iris1_test_data, w_iris1_test)
P_iris2_train, y_iris2_train = logistic_prediction(iris2_train_data, w_iris2_train)
P_iris2_test, y_iris2_test = logistic_prediction(iris2_test_data, w_iris2_test)


# Compute the number of errors by comparing the predicted classes and the real classes
errors1_train = np.sum(np.abs(y_iris1_train - iris1_train_labels)/2)
errors1_test = np.sum(np.abs(y_iris1_test - iris1_test_labels)/2)
errors2_train = np.sum(np.abs(y_iris2_train - iris2_train_labels)/2)
errors2_test = np.sum(np.abs(y_iris2_test - iris2_test_labels)/2)


# Compute the error rates
error_rate1_train = errors1_train/N_train1
error_rate1_test = errors1_test/N_test1
error_rate2_train = errors2_train/N_train2
error_rate2_test = errors2_test/N_train2

print "Error rate in train Iris2D1 dataset: %s" % round(error_rate1_train, 2) # 0.33
print "Error rate in test Iris2D1 dataset: %s" % round(error_rate1_test, 2)   # 0.17
print "Error rate in train Iris2D2 dataset: %s" % round(error_rate2_train, 2) # 0.29
print "Error rate in test Iris2D1 dataset: %s" % round(error_rate2_test, 2)   # 0.07


# Binary classification task in both training datasets

# Weights vector
w_IRIS1 = w_iris1_train

# IRIS 1 training dataset and y
x_IRIS1 = iris1_train_data
y_IRIS1 = -x_IRIS1 * (w_IRIS1[1]/w_IRIS1[2]) -w_IRIS1[0]/w_IRIS1[2]

# Limits of the binary classification using the weights obtain above
x1_IRIS1 = 4
x2_IRIS1 = 9

a_IRIS1 = -w_IRIS1[1] / w_IRIS1[2]
b_IRIS1 = -w_IRIS1[0] / w_IRIS1[2]

y1_IRIS1 = a_IRIS1 * x1_IRIS1 + b_IRIS1
y2_IRIS1 = a_IRIS1 * x2_IRIS1 + b_IRIS1


plt.plot([x1_IRIS1, x2_IRIS1], [y1_IRIS1, y2_IRIS1], color='r')
plt.scatter(x_IRIS1[:,0], x_IRIS1[:, 1])
plt.title('Binary classification of IRIS 1 dataset')

plt.show()

# Weights vector
w_IRIS2 = w_iris2_train

# IRIS 2 training dataset and y
x_IRIS2 = iris2_train_data
y_IRIS2 = -x_IRIS2 * (w_IRIS2[1]/w_IRIS2[2]) -w_IRIS2[0]/w_IRIS2[2]

# Limits of the binary classification using the weights obtain above
x1_IRIS2 = 4
x2_IRIS2 = 7

a_IRIS2 = -w_IRIS2[1] / w_IRIS2[2]
b_IRIS2 = -w_IRIS2[0] / w_IRIS2[2]

y1_IRIS2 = a_IRIS2 * x1_IRIS2 + b_IRIS2
y2_IRIS2 = a_IRIS2 * x2_IRIS2 + b_IRIS2


plt.plot([x1_IRIS2, x2_IRIS2], [y1_IRIS2, y2_IRIS2], color='r')
plt.scatter(x_IRIS2[:,0], x_IRIS2[:, 1])
plt.title('Binary classification of IRIS 2 dataset')

plt.show()





