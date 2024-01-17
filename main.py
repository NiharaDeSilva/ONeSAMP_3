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
from popSimulator import SimulatePopulations
import multiprocessing
import concurrent.futures
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import cross_val_predict

# import matplotlib.pyplot as plt
# from sklearn.metrics import PredictionErrorDisplay


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

# mutationRate = 0.000000012
mutationRate = 0.0012
if (args.r):
    mutationRate = float(args.r)

lowerNe = 4
if (args.lNe):
    lowerNe = int(args.lNe)

upperNe = 500
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

# rangeNe = "%d,%d" % (lowerNe, upperNe)
rangeNe = (lowerNe, upperNe)

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
#Result queue
results_list = []

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

rate = 0.0012
simulate_populations = SimulatePopulations()

# Generate random populations and calculate summary statistics
def processRandomPopulation(x):
    loci = inputFileStatistics.numLoci
    sampleSize = inputFileStatistics.sampleSize
    proc = multiprocessing.Process()
    process_id = os.getpid()
    # change the intermediate file name by process id
    intermediateFilename = str(process_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
    intermediateFile = os.path.join(path, intermediateFilename)
    # cmd = "%s -u%.9f -v%s -rC -l%d -i%d -d%s -s -t1 -b%s -f%f -o1 -p > %s" % (
    #     POPULATION_GENERATOR, mutationRate, rangeTheta, loci, sampleSize, rangeDuration, rangeNe, minAlleleFreq,
    #     intermediateFile)
    simulate_populations.generate_population_data(sampleSize, loci, rangeNe, mutationRate, intermediateFile)

    # if (DEBUG):
    #     print(cmd)
    #
    # returned_value = os.system(cmd)
    #
    # if returned_value:
    #     print("ERROR:main:Refactor did not run")
    # exit()
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

# allPopStats = "allPopStats_" + getName(fileName) + "_" + str(t)
# fileALLPOP = open(allPopStats, 'w+')

try:
    os.mkdir(path)
except FileExistsError:
    pass

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')
    # Parallel process the random populations and add to a list
    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        # As each task completes, put the result in the queue
        for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
            try:
                results_list.append(result)
            except Exception as e:
                print(f"Generated an exception: {e}")

# Write all population stats to a file to pass as input for Rscript
# with fileALLPOP as result_file:
#     for result in results_list:
#         result_file.write('\t'.join(result) + '\n')
#
try:
    shutil.rmtree(path, ignore_errors=True)
except FileExistsError:
    pass
# fileALLPOP.close()


#########################################
# FINISHING ALL POPULATIONS
########################################
# STARTING LINEAR REGRESSION
#########################################
# ALL_POP_STATS_FILE = allPopStats

# R SCRIPT
# rScriptCMD = "Rscript %s %s %s" % (FINAL_R_ANALYSIS, ALL_POP_STATS_FILE, inputPopStats)
# print(rScriptCMD)
# res = os.system(rScriptCMD)
#
# if (res):
#     print("ERROR:main: Could not run Rscript.  FATAL ERROR.")
#     exit()
#
# if (DEBUG):
#     print("Finish linear regression")
#
# print("--- %s seconds ---" % (time.time() - start_time))

################################
# LINEAR REGRESSION WITH SKLEARN
################################

# Assign input and all population stats to dataframes with column names
allPopStatistics = pd.DataFrame(results_list, columns=['Ne', 'Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])
inputStatsList = pd.DataFrame([inputStatsList], columns=['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])

# Assign dependent and independent variables for regression model
Z = np.array(inputStatsList[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium']])
X = np.array(allPopStatistics[['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium']])
y = np.array(allPopStatistics['Ne'])
y = np.array([float(value) for value in y if float(value) > 0])

# #Normalize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# Z_scaled = scaler.fit_transform(Z)
#
# #Apply box-cox transformation
# y_transformed, lambda_value = boxcox(y)

#Split train and test data for cross validation
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_transformed, test_size=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit the linear regression model
model = LinearRegression()
result = model.fit(X_train, y_train)

#Predict for Test values
y_pred = result.predict(X_test)

# Predict the value for the query point
# prediction = model.predict(Z_scaled)
prediction = model.predict(Z)
# y_original_scale = inv_boxcox(prediction, lambda_value)
# print("\n Effective population for input population:", y_original_scale[0])

####### CALCULATING CONFIDENCE INTERVAL #########

def getConfInterval(model, X_train, y_train, Z, prediction):
    # Convert to a numeric array
    X_train = X_train.astype(np.float64)
    y_train = y_train.astype(np.float64)
    Z = Z.astype(np.float64)

    # Predictions on the training data
    y_train_pred = model.predict(X_train)

    # Compute the MSE on the training data
    mse = np.mean((y_train - y_train_pred) ** 2)

    # Compute the standard error of the prediction
    # The standard error is sqrt((1/N) * (1 + new_X * (X'X)^-1 * new_X')) * sigma
    # Where sigma is the std deviation of the residuals (sqrt of MSE)

    # First, compute the (X'X)^-1 matrix
    XX_inv = np.linalg.inv(np.dot(X_train.T, X_train))

    # Then, compute the leverage (hat) matrix for the new data point
    hat_matrix = np.dot(np.dot(Z, XX_inv), Z.T)

    # Now calculate the standard error of the prediction
    std_error = np.sqrt((1 + hat_matrix) * mse)

    # The t value for the 95% confidence interval
    t_value = stats.t.ppf(1 - 0.05 / 2, df=len(X_train) - X_train.shape[1] - 1)

    # Confidence interval for the new prediction
    ci_lower = prediction - t_value * std_error
    ci_upper = prediction + t_value * std_error

    print(f"95% confidence interval: [{ci_lower[0][0].round(decimals=2)}, {ci_upper[0][0].round(decimals=2)}]")

# Output the result
print(f"\nPrediction of Linear Regression Model: {prediction.round(decimals=2)}")
getConfInterval(model, X_train, y_train, Z, prediction)

# Get the coefficients for each feature
coefficients = model.coef_
# coefficients_original_scale = coefficients / lambda_value

# Print the coefficients for each feature
print("\nCoefficients for each feature:")
for feature, coef in zip(inputStatsList.columns, coefficients):
    print(f"Variable: {feature}: {coef:.2f}")


# Perform k-fold cross-validation
# cv_scores = cross_val_score(model, X_scaled, y_transformed, cv=10)
# cv_scores = cross_val_score(model, X, y, cv=10)
# print("\nCross validation scores : ", round(cv_scores[0],2), round(cv_scores[1],2), round(cv_scores[2],2), round(cv_scores[3],2), round(cv_scores[4],2))


print("--- %s seconds ---" % (time.time() - start_time))



# Deleting temporary files
# delete1 = "rm " + inputPopStats
# delete_INPUTPOP = os.system(delete1)
#
# delete2 = "rm " + allPopStats
# delete_ALLPOP = os.system(delete2)
##########################
# RANDOM FOREST REGRESSION
##########################
# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on test data
y_pred = rf_regressor.predict(X_test)

# Predict the Ne value for input population
rf_prediction = rf_regressor.predict(Z)
print(f"\nPrediction of Random Forest Regression Model: {rf_prediction.round(decimals=2)}")

# Calculate confidence interval
# Get the predictions from each tree for the new data point
tree_predictions = np.array([tree.predict(Z) for tree in rf_regressor.estimators_])

# Calculate the 2.5th and 97.5th percentiles for the 95% confidence interval
lower_bound = np.percentile(tree_predictions, 2.5)
upper_bound = np.percentile(tree_predictions, 97.5)

# Output the confidence interval
print(f"95% confidence interval for the new data point: [{lower_bound.round(decimals=2)}, {upper_bound.round(decimals=2)}]")


# Using Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Get numerical feature importances
importances = list(rf_regressor.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(inputStatsList.columns, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print out the feature and importances
print("\nFeature importance")
[print('Variable: {:30} Importance: {}'.format(*pair)) for pair in feature_importances]


print("--- %s seconds ---" % (time.time() - start_time))


##########################
# END
#########################
# FNN
#########################
import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.preprocessing import StandardScaler

# Standardizing data
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = X_train.astype(np.float32)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = y_train.astype(np.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = X_test.astype(np.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = y_test.astype(np.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100   # number of epochs to run
batch_size = 10  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
# plt.plot(history)
# plt.show()

# model.eval()
# with torch.no_grad():
#     # Test out inference with 5 samples
#     for i in range(5):
#         X_sample = X_test[i: i+1]
#         # X_sample = scaler.transform(X_sample)
#         X_sample = torch.tensor(X_sample, dtype=torch.float32)
#         y_pred = model(X_sample)
#         print(f"{X_test[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")


model.eval()  # Set the model to evaluation mode
Z = Z.astype(np.float32)
with torch.no_grad():
    input_data = torch.tensor(Z, dtype=torch.float32)  # Convert new data to a PyTorch tensor
    prediction = model(input_data)  # Forward pass to make predictions
    predicted_value = prediction.item()  # Extract the predicted value (assuming a single output)

print(f"\nPrediction of Feedforward Neural Network Model: {round(predicted_value,2)}")

