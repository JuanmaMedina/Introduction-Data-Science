from __future__ import division
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

# CSV files with the train and test datasets
dataTrain = np.loadtxt('/home/juanma/IDS/IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('/home/juanma/IDS/IDSWeedCropTest.csv', delimiter=',')

# 4 arrays with input variables and labels of train and test datasets respectively
XTrain = dataTrain[:, :-1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, :-1]
YTest = dataTest[:, -1]


# EXERCISE 1: Apply a 1-NN classifier to the data and determine the classification accuracy

def custom_classifier(trainX, testX, trainY, testY):
    '''Custom 1-NN classifier that trains with a train dataset and predicts labels of a test dataset with an accuracy'''

    # List of predicted labels
    labels = []

    # Predict the label of each data in training dataset
    for data1 in testX:

        # List of distances between test and train data
        distances = []

        # Compute the vectorial distance between every data in test and train sets and add them to distances list
        for data2 in trainX:
            dist = np.linalg.norm(data1 - data2)
            distances.append(dist)

        # Extract index of the minimum distance, get label of the train data with that index and add it to labels list
        pred_index = distances.index(min(distances))
        labels.append(trainY[pred_index])

    score = 0
    # For every predicted label in the list, if it is the same as the label of the test dataset, sum 1 to the counter
    for label in range(len(labels)):
        if labels[label] == testY[label]:
            score += 1

    # Calculate and return the accuracy of the classifier
    return (score / len(testY))

print custom_classifier(XTrain, XTest, YTrain, YTest) # 0.945993031359


# Set-up classifier with built-in function and train it using XTrain as training data and YTrain as labels
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(XTrain, YTrain)

# Determine the model accuracy on the given test dataset by comparing the real labels VS model predicted labels
accTest = accuracy_score(YTest, clf.predict(XTest)) # 0.945993031359


# EXERCISE 2: Find the best odd k value (from 1 to 25) estimating the performance of each classifier using 5-fold CV

# Set a 5-fold CV
cv = KFold(n_splits=5)

# Division of XTrain in 5 sets with 1000 data, each with a training dataset (4/5 = 800) and a test dataset (1/5 = 200)
for train, test in cv.split(XTrain):
    XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train], XTrain[test], YTrain[train], YTrain[test]

    # Empty list where k values and their accuracies will be appended
    scores_k = []

    # Set a classifier for each subset, train it and determine its accuracy for each odd K
    for k in range(1, 25, 2):
        cm_clf = KNeighborsClassifier(n_neighbors=k)
        cm_clf.fit(XTrainCV, YTrainCV)

        cm_accTest = accuracy_score(YTestCV, cm_clf.predict(XTestCV))

        # Add each k value and its accuracy to the scores_k list as a tuple
        scores_k.append((cm_accTest, k))

# print max(scores_k) # (0.96999999999999997, 3); (0.96999999999999997, 5)


# EXERCISE 3: Apply the Kbest-NN classifier to the data and determine the classification accuracy

# Same process as Exercise 1 with the Kbest estimator found in Exercise 2 (k = 3)
clf_best = KNeighborsClassifier(n_neighbors=3)
clf_best.fit(XTrain, YTrain)
accTest_best = accuracy_score(YTest, clf_best.predict(XTest))

print accTest_best # 0.949477351916


# EXERCISE 4: Center and normalize the data and repeat Exercises 2 and 3

# Custom standardization of the data
feature_means_Train = np.mean(XTrain, axis=0)
feature_var_Train = np.var(XTrain, axis=0)
feature_sd_Train = np.sqrt(feature_var_Train)

feature_means_Test = np.mean(XTest, axis=0)
feature_var_Test = np.var(XTest, axis=0)
feature_sd_Test = np.sqrt(feature_var_Test)

XTrainN_custom = (XTrain - feature_means_Train) / feature_sd_Train
XTestN_custom = (XTest - feature_means_Test) / feature_sd_Test

# Same CV process and model selection as in Exercise 2, but this time basing it on the custom-standardized trainset
for train, test in cv.split(XTrainN_custom):
    XTrainCVN, XTestCVN, YTrainCVN, YTestCVN = XTrainN_custom[train], XTrainN_custom[test], YTrain[train], YTrain[test]

    scores_k_N_custom = []

    for k in range(1, 25, 2):
        cm_clf_N = KNeighborsClassifier(n_neighbors=k)
        cm_clf_N.fit(XTrainCVN, YTrainCVN)

        cm_accTest_N = accuracy_score(YTestCVN, cm_clf_N.predict(XTestCVN))

        scores_k_N_custom.append((cm_accTest_N, k))

# print scores_k_N_custom # (0.97499999999999998, 3); (0.97499999999999998, 5); (0.97499999999999998, 7)


# Built-in standardization: creation of the scaler object
scaler = preprocessing.StandardScaler().fit(XTrain)

# Apply this scaling to both training and test datasets
XTrainN = scaler.transform(XTrain)
XTestN = scaler.transform(XTest)

# Same CV process and model selection as in Exercise 2, but this time basing it on the built-in-standardized trainset
for train, test in cv.split(XTrainN):
    XTrainCVN, XTestCVN, YTrainCVN, YTestCVN = XTrainN[train], XTrainN[test], YTrain[train], YTrain[test]

    scores_k_N = []

    for k in range(1, 25, 2):
        cm_clf_N = KNeighborsClassifier(n_neighbors=k)
        cm_clf_N.fit(XTrainCVN, YTrainCVN)

        cm_accTest_N = accuracy_score(YTestCVN, cm_clf_N.predict(XTestCVN))

        scores_k_N.append((cm_accTest_N, k))

# print scores_k_N # (0.97499999999999998, 3); (0.97499999999999998, 5); (0.97499999999999998, 7)


# Same process as Exercise 1 with the KbestN estimator found above (k = 3)
clf_best_N = KNeighborsClassifier(n_neighbors=3)
clf_best_N.fit(XTrainN, YTrain)
accTest_best_N = accuracy_score(YTest, clf_best_N.predict(XTestN))

print accTest_best_N # 0.959930313589

