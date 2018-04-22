from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import pearsonr


# This file contains the datamatrix of the smoking dataset
file = np.loadtxt('/home/juanma/IDS/smoking.txt')

# EXERCISE 1: Compute the average lung function (in FEV1) among the smokers and non-smokers (using "meanFEV1.py")

# Lists to store the FEV1 values of smokers and non-smokers
smokers = []
non_smokers = []

# For each child, if smoker (1.0), append his FEV1 value to smokers list. If not (0.0), append it to non_smokers.
for child in file:
    if child[4] == 1.0:
        smokers.append(child[1])
    else:
        non_smokers.append(child[1])

def meanFEV1(data):
    """This function returns the average mean value FEV1 of smokers and nonsmokers"""
    FEV1 = sum(data) / len(data) # It could also be computed directly with the np.mean function
    return FEV1

print (meanFEV1(smokers), meanFEV1(non_smokers)) # (3.277, 2.566)


# EXERCISE 2: Make a boxplot of the FEV1 values in the two groups

data_box = [smokers, non_smokers]

labels = ['Smokers', 'Non-smokers']
plt.boxplot(data_box, labels=labels)
plt.title('Boxplot over FEV1 values for smokers and non-smokers')
plt.ylabel('FEV1 value')

plt.show()


# EXERCISE 3: Analyze the difference in the FEV1 levels of smokers and non-smokers with a two-sided T test

# Number of smokers and non-smokers
n_smk = len(smokers)
n_nosmk = len(non_smokers)

# Variances of smokers and non-smokers datasets
v_smk = np.var(smokers)
v_nosmk = np.var(non_smokers)

# Computing the two-sample T statistic (formula) # 7.199
t_for = (meanFEV1(smokers) - meanFEV1(non_smokers)) / np.sqrt((v_smk / n_smk) + (v_nosmk / n_nosmk))

# Computing the degrees of freedom (formula), rounding to nearest integer # 83
v_for = int((((v_smk / n_smk) + (v_nosmk / n_nosmk)) ** 2) /
            (((v_smk ** 2) / ((n_smk ** 2) * (n_smk - 1))) + ((v_nosmk ** 2) / ((n_nosmk ** 2) * (n_nosmk - 1)))))

def hyptest(t_stat, v):
    """This function returns True if H0 (equal means) is rejected (p < 0.05) and False otherwise"""
    p = 2 * t.cdf(- t_stat, v)
    # To check the p-value (2.495e-10)
    # print p
    if p < 0.05:
        return True
    else:
        return False

print hyptest(t_for, v_for)


# EXERCISE 4: Make a 2D plot of age VS FEV1 and compute the correlation between them

age = file[:,0]
FEV1 = file[:,1]

plt.scatter(age, FEV1, label='Age VS FEV1')
plt.title('Age VS FEV1')
plt.xlabel('Age')
plt.ylabel('FEV1')

plt.show()

# Computation of the correlation with the scipy given formula
corr = pearsonr(age, FEV1)

print corr # 0.756


# EXERCISE 5: Create a histogram over the age of the subjects in the smokers and non-smokers groups

# Lists to store the ages of smokers and non-smokers
age_smokers = []
age_non_smokers = []

# For each child, if smoker (1.0), append his age to age_smokers list. If not (0.0), append it to age_non_smokers.
# This loop could be added to the first one to save effort in computation terms
for child in file:
    if child[4] == 1.0:
        age_smokers.append(child[0])
    else:
        age_non_smokers.append(child[0])

plt.hist(age_smokers, 22)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram over the age of the smokers')

plt.show()

plt.hist(age_non_smokers, 33)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Histogram over the age of the non-smokers')

plt.show()
