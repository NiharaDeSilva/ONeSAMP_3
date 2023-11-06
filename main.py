#!/usr/bin/python
import subprocess
import sys
import argparse

import os
import shutil
import numpy as np
import pandas as pd
import time
from statistics import statisticsClass
import multiprocessing
import concurrent.futures
import warnings
from sklearn.linear_model import LinearRegression
from scipy import stats

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt
from sklearn.metrics import PredictionErrorDisplay


NUMBER_OF_STATISTICS = 5
t = 30
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
OUTPUTFILENAME = "priors.txt"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
directory = "temp"
path = os.path.join("./", directory)

POPULATION_GENERATOR = "./build/OneSamp"
FINAL_R_ANALYSIS = "./scripts/rScript.r"


def getName(filename):
    (_, filename) = os.path.split(filename)
    return filename


#############################################################
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=float, help="Minimum Allele Frequency")
parser.add_argument("--r", type=float, help="Mutation Rate")
parser.add_argument("--lNe", type=int, help="Lower of Ne Range")
parser.add_argument("--uNe", type=int, help="Upper of Ne Range")
parser.add_argument("--lT", type=float, help="Lower of Theta Range")
parser.add_argument("--uT", type=float, help="Upper of Theta Range")
parser.add_argument("--s", type=int, help="Number of OneSamp Trials")
parser.add_argument("--lD", type=float, help="Lower of Duration Range")
parser.add_argument("--uD", type=float, help="Upper of Duration Range")
parser.add_argument("--i", type=float, help="Missing data for individuals")
parser.add_argument("--l", type=float, help="Missing data for loci")
parser.add_argument("--o", type=str, help="The File Name")
parser.add_argument("--t", type=int, help="Repeat times")
parser.add_argument("--n", type=bool, help="whether to filter the monomorphic loci", default=False)

args = parser.parse_args()

#########################################
# INITIALIZING PARAMETERS
#########################################
if (args.t):
    t = int(args.t)

minAlleleFreq = 0.05
if (args.m):
    minAlleleFreq = float(args.m)

mutationRate = 0.000000012
if (args.r):
    mutationRate = float(args.r)

lowerNe = 150
if (args.lNe):
    lowerNe = int(args.lNe)

upperNe = 250
if (args.uNe):
    upperNe = int(args.uNe)

if (int(lowerNe) > int(upperNe)):
    print("ERROR:main:lowerNe > upperNe. Fatal Error")
    exit()

if (int(lowerNe) < 1):
    print("ERROR:main:lowerNe must be a positive value. Fatal Error")
    exit()

if (int(upperNe) < 1):
    print("ERROR:main:upperNe must be a positive value. Fatal Error")
    exit()

rangeNe = "%d,%d" % (lowerNe, upperNe)

lowerTheta = 0.000048
if (args.lT):
    lowerTheta = float(args.lT)

upperTheta = 0.0048
if (args.uT):
    upperTheta = float(args.uT)

rangeTheta = "%f,%f" % (lowerTheta, upperTheta)

numOneSampTrials = 50000
if (args.s):
    numOneSampTrials = int(args.s)

lowerDuration = 2
if (args.lD):
    lowerDuration = float(args.lD)

upperDuration = 8
if (args.uD):
    upperDuration = float(args.uD)

indivMissing = .2
if (args.i):
    indivMissing = float(args.i)

lociMissing = .2
if (args.l):
    lociMissing = float(args.l)

rangeDuration = "%f,%f" % (lowerDuration, upperDuration)

fileName = "oneSampIn"
if (args.o):
    fileName = str(args.o)
else:
    print("WARNING:main: No filename provided.  Using oneSampIn")

if (DEBUG):
    print("Start calculation of statistics for input population")

rangeTheta = "%f,%f" % (lowerTheta, upperTheta)

#########################################
# STARTING INITIAL POPULATION
#########################################

inputFileStatistics = statisticsClass()

# t = time.time()
inputFileStatistics.readData(fileName)
inputFileStatistics.filterIndividuals(indivMissing)
inputFileStatistics.filterLoci(lociMissing)
if (args.n):
    inputFileStatistics.filterMonomorphicLoci()
inputFileStatistics.test_stat1()
inputFileStatistics.test_stat2()
inputFileStatistics.test_stat3()
inputFileStatistics.test_stat5()
inputFileStatistics.test_stat4()
# print(f'coast:{time.time() - t:.4f}s')
#
# t = time.time()
# inputFileStatistics.testRead(fileName)
# # inputFileStatistics.filterIndividuals(indivMissing)
# # inputFileStatistics.filterLoci(lociMissing)
# inputFileStatistics.new_stat1()
# inputFileStatistics.stat2()
# inputFileStatistics.stat3()
# inputFileStatistics.newStat4()
#
# inputFileStatistics.stat5()
# print(f'coast:{time.time() - t:.4f}s')
numLoci = inputFileStatistics.numLoci
sampleSize = inputFileStatistics.sampleSize

##Creating input file & List with intial statistics
textList = [str(inputFileStatistics.stat1), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
            str(inputFileStatistics.stat4), str(inputFileStatistics.stat5)]
inputStatsList = textList

inputPopStats = "inputPopStats_" + getName(fileName) + "_" + str(t)
with open(inputPopStats, 'w') as fileINPUT:
    fileINPUT.write('\t'.join(textList[0:]) + '\t')
fileINPUT.close()

if (DEBUG):
    print("Finish calculation of statistics for input population")

#############################################
# FINISH STATS FOR INITIAL INPUT  POPULATION
############################################

#########################################
# STARTING ALL POPULATIONS
#########################################

if (DEBUG):
    print("Start calculation of statistics for ALL populations")

statistics1 = []
statistics2 = []
statistics3 = []
statistics4 = []
statistics5 = []

statistics1 = [0 for x in range(numOneSampTrials)]
statistics2 = [0 for x in range(numOneSampTrials)]
statistics3 = [0 for x in range(numOneSampTrials)]
statistics5 = [0 for x in range(numOneSampTrials)]
statistics4 = [0 for x in range(numOneSampTrials)]


# Generate random populations and calculate summary statistics
def processRandomPopulation(x):
    loci = inputFileStatistics.numLoci
    sampleSize = inputFileStatistics.sampleSize
    proc = multiprocessing.Process()
    process_id = os.getpid()
    # change the intermediate file name by thread id
    intermediateFilename = str(process_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
    intermediateFile = os.path.join(path, intermediateFilename)
    cmd = "%s -u%.9f -v%s -rC -l%d -i%d -d%s -s -t1 -b%s -f%f -o1 -p > %s" % (
        POPULATION_GENERATOR, mutationRate, rangeTheta, loci, sampleSize, rangeDuration, rangeNe, minAlleleFreq,
        intermediateFile)

    if (DEBUG):
        print(cmd)

    returned_value = os.system(cmd)

    if returned_value:
        print("ERROR:main:Refactor did not run")
        exit()

    refactorFileStatistics = statisticsClass()

    refactorFileStatistics.readData(intermediateFile)
    refactorFileStatistics.test_stat1()
    refactorFileStatistics.test_stat2()
    refactorFileStatistics.test_stat3()
    refactorFileStatistics.test_stat5()
    refactorFileStatistics.test_stat4()

    statistics1[x] = refactorFileStatistics.stat1
    statistics2[x] = refactorFileStatistics.stat2
    statistics3[x] = refactorFileStatistics.stat3
    statistics5[x] = refactorFileStatistics.stat5
    statistics4[x] = refactorFileStatistics.stat4

    # Making file with stats from all populations
    textList = []
    textList = [str(refactorFileStatistics.NE_VALUE), str(refactorFileStatistics.stat1),
                str(refactorFileStatistics.stat2),
                str(refactorFileStatistics.stat3),
                str(refactorFileStatistics.stat4), str(refactorFileStatistics.stat5)]
    return textList

allPopStats = "allPopStats_" + getName(fileName) + "_" + str(t)
fileALLPOP = open(allPopStats, 'w+')

try:
    os.mkdir(path)
except FileExistsError:
    pass

# Worker function that computes statistics and puts the result in a queue

#Result queue
manager = multiprocessing.Manager()
result_queue = manager.Queue()
results_list = []

# Parallel process the random populations and add to a queue/list
with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
    # As each task completes, put the result in the queue
    for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
        result_queue.put(result)
        results_list.append(result)

# Write all population stats to a file to pass as a input to R script
with fileALLPOP as result_file:
    while not result_queue.empty():
        result = result_queue.get()
        result_file.write('\t'.join(result) + '\n')

allPopStatistics = pd.DataFrame(results_list, columns=['Ne', 'Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])

try:
    shutil.rmtree(path, ignore_errors=True)
except FileExistsError:
    pass
fileALLPOP.close()

cal_time = time.time()
print(cal_time)

#########################################
# FINISHING ALL POPULATIONS
########################################
# STARTING LINEAR REGRESSION
#########################################
ALL_POP_STATS_FILE = allPopStats

# R SCRIPT
rScriptCMD = "Rscript %s %s %s" % (FINAL_R_ANALYSIS, ALL_POP_STATS_FILE, inputPopStats)
print(rScriptCMD)
res = os.system(rScriptCMD)

if (res):
    print("ERROR:main: Could not run Rscript.  FATAL ERROR.")
    exit()

if (DEBUG):
    print("Finish linear regression")

print("--- %s seconds ---" % (time.time() - start_time))


# TODO double check there
inputStatsList = pd.DataFrame([inputStatsList], columns=['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])

Z = np.array(inputStatsList[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium']])
X = np.array(allPopStatistics[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium']])
y = np.array(allPopStatistics['Ne'])
y = np.array([float(value) for value in y if float(value) > 0])

#Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Z_scaled = scaler.fit_transform(Z)

#Apply box-cox transformation
y_transformed, lambda_value = boxcox(y)

#Split train and test data for cross validation
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_transformed, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fit the linear regression model
model = LinearRegression()
# model.fit(X_train, y_train, sample_weight=weights)
result = model.fit(X_train, y_train)

#Predict for Test values
y_pred = result.predict(X_test)

# Perform k-fold cross-validation
# Model evaluation with cross validation
# print("Model Evaluation\n")
# cv_scores = cross_val_score(model, X_scaled, y_transformed, cv=10)
# cv_pred = cross_val_predict(model, X_scaled, y_transformed, cv=10)
cv_scores = cross_val_score(model, X, y, cv=10)
cv_pred = cross_val_predict(model, X, y, cv=10)
print("Cross validation scores : ", cv_scores[0], cv_scores[1], cv_scores[2], cv_scores[3], cv_scores[4])

# Get the coefficients for each feature
coefficients = model.coef_
coefficients_original_scale = coefficients / lambda_value

# Print the coefficients for each feature
print("\nCoefficients for each feature:")
for feature, coef in zip(inputStatsList.columns, coefficients_original_scale):
    print(f"{feature}: {coef:.4f}")

# Predict the value for the query point
# prediction = model.predict(Z_scaled)
prediction = model.predict(Z)
y_original_scale = inv_boxcox(prediction, lambda_value)
# print("\n Effective population for input population:", y_original_scale[0])
print("\n Effective population for input population:", prediction)


# for 95% confidence interval; use 0.01 for 99%-CI.
alpha = 0.05

# import statsmodels.api as sm
# alpha = 0.05 # 95% confidence interval
# lr = sm.OLS(y_train, sm.add_constant(X_train)).fit()
# conf_interval = lr.conf_int(alpha)
# print(conf_interval)


def get_conf_int(alpha, lr, X, y):
    # Ensure X is a NumPy array
    X = np.array(X)
    X = X.astype(float)
    y = np.array(y)

    # Add a column of ones to the features matrix to represent the intercept
    X_aux = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    # Calculate degrees of freedom
    dof = X_aux.shape[0] - X_aux.shape[1]

    # Calculate the mean squared error (mse)
    mse = np.sum((y - lr.predict(X)) ** 2) / dof

    # Calculate variance of parameters
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))

    # Get t-statistic value for alpha/2
    t_val = stats.t.ppf(1 - alpha / 2, dof)

    # Calculate the margin of error (gap)
    gap = t_val * np.sqrt(mse * var_params)

    # Coefficients, including intercept
    coefs = np.concatenate([[lr.intercept_], lr.coef_])

    # Create lower and upper bounds
    lower_bound = coefs - gap
    upper_bound = coefs + gap

    # You can return the bounds in whatever format you prefer;
    # here we'll stack them into an array for convenience.
    return pd.DataFrame({
        'lower': coefs - gap, 'upper': coefs + gap
    })

conf_interval = get_conf_int(alpha, model, X_train, y_train)
print(conf_interval)



# # Calculate R2-Score
# r2_score = r2_score(y_test, y_pred)
# print(f'R2_Score: {r2_score}')

# #Plot prediction errors
# fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
# PredictionErrorDisplay.from_predictions(
#     y,
#     y_pred=cv_pred,
#     kind="actual_vs_predicted",
#     subsample=100,
#     ax=axs[0],
#     random_state=0,
# )
# axs[0].set_title("Actual vs. Predicted values")
# PredictionErrorDisplay.from_predictions(
#     y,
#     y_pred=cv_pred,
#     kind="residual_vs_predicted",
#     subsample=100,
#     ax=axs[1],
#     random_state=0,
# )
# axs[1].set_title("Residuals vs. Predicted Values")
# fig.suptitle("Plotting cross-validated predictions")
# plt.tight_layout()
# plt.show()


print("--- %s seconds ---" % (time.time() - start_time))


# Deleting temporary files
# delete1 = "rm " + inputPopStats
# delete_INPUTPOP = os.system(delete1)
#
# delete2 = "rm " + allPopStats
# delete_ALLPOP = os.system(delete2)

##########################
# END
#########################
