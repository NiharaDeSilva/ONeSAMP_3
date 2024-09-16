#!/usr/bin/python
import subprocess
import sys
import argparse

import os
import shutil
import numpy as np
import time

from joblib import dump
#from popSimulator import SimulatePopulations
from statistics import statisticsClass
import pandas as pd
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
import random


import multiprocessing
import concurrent.futures
import warnings

# import matplotlib.pyplot as plt
# from sklearn.metrics import PredictionErrorDisplay
import subprocess

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
parser.add_argument("--md", type=str, help="Model Name")

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
# mutationRate = 0.012

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
# rangeNe = (lowerNe, upperNe)

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

modelName = "ridge.joblib"
if (args.md):
    modelName = str(args.md)
else:
    print("WARNING:main: No filename provided.  Using ridge.joblib")

if (DEBUG):
    print("Start calculation of statistics for input population")

rangeTheta = "%f,%f" % (lowerTheta, upperTheta)
duration_start=2
duration_range=6
missing_data_percentage=0.2


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

numLoci = inputFileStatistics.numLoci
sampleSize = inputFileStatistics.sampleSize

##Creating input file & List with intial statistics
textList = [str(inputFileStatistics.stat1), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
             str(inputFileStatistics.stat4), str(inputFileStatistics.stat5)]
#textList = [str(inputFileStatistics.stat1), str(inputFileStatistics.stat3),
#            str(inputFileStatistics.stat4)]
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

# File for all population stats
# allPopStats = "allPopStats_" + getName(fileName) + "_" + str(t)
# fileALLPOP = open(allPopStats, 'w+')

# Generate random populations and calculate summary statistics
def processRandomPopulation(x):
    loci = inputFileStatistics.numLoci
    sampleSize = inputFileStatistics.sampleSize
    proc = multiprocessing.Process()
    process_id = os.getpid()
    # change the intermediate file name by process id
    intermediateFilename = str(process_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
    intermediateFile = os.path.join(path, intermediateFilename)
    Ne_left = lowerNe
    Ne_right = upperNe
    if Ne_left % 2 != 0:
        Ne_left += 1
    num_evens = (Ne_right - Ne_left) // 2 + 1
    random_index = random.randint(0, num_evens - 1)
    target_Ne = Ne_left + random_index * 2
    target_Ne = f"{target_Ne:05d}"
    #cmd = "%s -u%.9f -v%s -rC -l%d -i%d -d%s -s -t1 -b%s -f%f -o1 -p > %s" % (POPULATION_GENERATOR, mutationRate, rangeTheta, loci, sampleSize, rangeDuration, target_Ne, minAlleleFreq, intermediateFile)
    simulate_populations = SimulatePopulations()
    simulate_populations.generate_population_data(sampleSize, loci, rangeNe, mutationRate, intermediateFile, duration_start, duration_range, missing_data_percentage)

    '''
    if (DEBUG):
        print(cmd)

    returned_value = os.system(cmd)

    if returned_value:
        print("ERROR:main:Refactor did not run")
    '''

    refactorFileStatistics = statisticsClass()

    refactorFileStatistics.readData(intermediateFile)
    refactorFileStatistics.filterIndividuals(indivMissing)
    refactorFileStatistics.filterLoci(lociMissing)
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
    textList = [str(refactorFileStatistics.NE_VALUE), str(refactorFileStatistics.stat1),
                str(refactorFileStatistics.stat2),
                str(refactorFileStatistics.stat3),
                str(refactorFileStatistics.stat4), str(refactorFileStatistics.stat5)]

    return textList


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

#
# # Write all population stats to a file to pass as input for Rscript
# with fileALLPOP as result_file:
#     for result in results_list:
#         result_file.write('\t'.join(result) + '\n')
#
# ALL_POP_STATS_FILE = allPopStats
#
# try:
#    shutil.rmtree(path, ignore_errors=True)
# except FileExistsError:
#    pass
# fileALLPOP.close()


#########################################
# FINISHING ALL POPULATIONS
########################################
# STARTING LINEAR REGRESSION
#########################################
'''
rScriptCMD = "Rscript %s %s %s" % (FINAL_R_ANALYSIS, ALL_POP_STATS_FILE, inputPopStats)

res = os.system(rScriptCMD)

if (res):
     print("ERROR:main: Could not run Rscript.  FATAL ERROR.")
     exit()

if (DEBUG):
    print("Finish linear regression")

print("--- %s seconds ---" % (time.time() - start_time))
'''
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

#
# Fit the linear regression model
model = LinearRegression()
result = model.fit(X_train, y_train)

print(f"\n-----------------LINEAR REGRESSION------------------")

#Predict for Test values
y_pred = result.predict(X_test)

# Calculate errors
absolute_errors = np.abs(y_pred - y_test)
min = np.min(absolute_errors)
max = np.max(absolute_errors)
q1 = np.percentile(absolute_errors, 25)
median = np.percentile(absolute_errors, 50)
q3 = np.percentile(absolute_errors, 75)
mae = np.mean(absolute_errors)

# Compute MSE, RMSE, and MAE for the test set
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"MAE: {mae_test:.2f}")
print(f"{min:.2f} {max:.2f} {median:.2f} {q1:.2f} {q3:.2f}")

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
    # MSE on the training data
    mse = np.mean((y_train - y_train_pred) ** 2)
    # compute the (X'X)^-1 matrix
    XX_inv = np.linalg.inv(np.dot(X_train.T, X_train))
    # compute the leverage (hat) matrix for the new data point
    hat_matrix = np.dot(np.dot(Z, XX_inv), Z.T)
    # calculate the standard error of the prediction
    std_error = np.sqrt((1 + hat_matrix) * mse)
    # The t value for the 95% confidence interval
    t_value = stats.t.ppf(1 - 0.05 / 2, df=len(X_train) - X_train.shape[1] - 1)

    # Confidence interval for the new prediction
    ci_lower = prediction - t_value * std_error
    ci_upper = prediction + t_value * std_error

    print(f"95% confidence interval: [{ci_lower[0][0].round(decimals=2)}, {ci_upper[0][0].round(decimals=2)}]")

# Output the result
print(f"\nPrediction: {prediction.round(decimals=2)}")
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

print("----- %s seconds -----" % (time.time() - start_time))


# Deleting temporary files
# delete1 = "rm " + inputPopStats
# delete_INPUTPOP = os.system(delete1)

# # delete2 = "rm " + allPopStats
# # delete_ALLPOP = os.system(delete2)
# ##########################
# # RANDOM FOREST REGRESSION
# ##########################
# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=80, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

print(f"\n-----------------RANDOM FOREST------------------")

# Make predictions on test data
y_pred = rf_regressor.predict(X_test)

# Calculate errors
absolute_errors = np.abs(y_pred - y_test)
min = np.min(absolute_errors)
max = np.max(absolute_errors)
q1 = np.percentile(absolute_errors, 25)
median = np.percentile(absolute_errors, 50)
q3 = np.percentile(absolute_errors, 75)
mae = np.mean(absolute_errors)

# Predict the Ne value for input population
rf_prediction = rf_regressor.predict(Z)

# Using Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"{min:.2f} {max:.2f} {median:.2f} {q1:.2f} {q3:.2f}")


print(f"\nPrediction:")
print(rf_prediction.round(decimals=2))

# Calculate confidence interval
# Get the predictions from each tree for the new data point
tree_predictions = np.array([tree.predict(Z) for tree in rf_regressor.estimators_])

# Calculate median
median_prediction = np.median(tree_predictions, axis=0)
print(f"\nMedian Prediction: ")
print(rf_prediction.round(decimals=2))

# Calculate the 2.5th and 97.5th percentiles for the 95% confidence interval
lower_bound = np.percentile(tree_predictions, 2.5)
upper_bound = np.percentile(tree_predictions, 97.5)

# Output the confidence interval
print(f"95% confidence interval:")
print(f"{lower_bound.round(decimals=2), upper_bound.round(decimals=2)}")

# Get numerical feature importances
importances = list(rf_regressor.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(inputStatsList.columns, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# Print out the feature and importances
print("\nFeature importance")
[print('Variable: {:30} : {}'.format(*pair)) for pair in feature_importances]


print("----- %s seconds -----" % (time.time() - start_time))

#
# ##########################
# # END
# #########################
# # FNN
# #########################
import copy
#from skorch import NeuralNetRegressor
#from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Convert to PyTorch tensors
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# train-test split for model evaluation
# X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Standardizing data
# scaler = StandardScaler()
# scaler.fit(X_train_raw)
# X_train = scaler.transform(X_train_raw)
# X_test = scaler.transform(X_test_raw)


X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = y_train.astype(np.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
y_test = y_test.astype(np.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)
Z = Z.astype(np.float32)
Z = torch.tensor(Z, dtype=torch.float32).to(device)

# # Convert to 2D PyTorch tensors
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(5, 100),
    nn.ReLU(),
    nn.Linear(100, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
model.to(device)

# loss function and optimizer
loss_fn = nn.MSELoss()  # mean square error
optimizer = optim.Adam(model.parameters(), lr=0.01)

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

print(f"\n-----------------NEURAL NETWORK------------------")

# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))

# Convert tensors to numpy arrays
y_test = y_test.numpy()
y_pred = y_pred.detach().numpy()  # Ensure y_pred is detached from the computation graph

# Calculate absolute errors
absolute_errors = np.abs(y_pred - y_test)
min = np.min(absolute_errors)
max = np.max(absolute_errors)
q1 = np.percentile(absolute_errors, 25)
median = np.percentile(absolute_errors, 50)
q3 = np.percentile(absolute_errors, 75)
mae = np.mean(absolute_errors)
print(f"MAE: {mae:.2f}")
print(f"{min:.2f} {max:.2f} {median:.2f} {q1:.2f} {q3:.2f}")
# ##########################
#plt.plot(history)
#plt.show()

model.eval()

# Number of simulations
n_simulations = 100
# Array to store predictions
predictions = np.zeros(n_simulations)

# Standard deviation of noise to add to Z for simulations
# Adjust the scale based on your expected input variability
noise_std = 0.01 * torch.std(Z)

for i in range(n_simulations):
    # Add random noise to Z
    random = torch.randn(Z.shape)
    Z_perturbed = Z + (random * noise_std)
    # Predict with model
    with torch.no_grad():
        pred = model(Z_perturbed)
    # Ensure pred is converted to a scalar if necessary, assuming pred should be a single value
    pred_scalar = pred.numpy().flatten()[0]  # Flatten and take the first element to ensure scalar conversion
    predictions[i] = pred_scalar

# Calculate confidence interval
lower = np.percentile(predictions, 2.5)
upper = np.percentile(predictions, 97.5)

print(f"Neural network prediction: ")
print(f"{np.mean(predictions).round(2)}")
print(f"median prediction: {np.median(predictions).round(2)}")
print(f"95% confidence interval:")
print(f"{lower:.2f}, {upper:.2f}")

print("----- %s seconds -----" % (time.time() - start_time))






# Assign input and all population stats to dataframes with column names
# allPopStatistics = pd.DataFrame(results_list, columns=['Ne', 'Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])
# inputStatsList = pd.DataFrame([inputStatsList], columns=['Emean_exhyt', 'Fix_index', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Gametic_disequilibrium'])
#
# Z = np.array(inputStatsList[['Emean_exhyt','Fix_index','Mlocus_homozegosity_mean','Mlocus_homozegosity_variance','Gametic_disequilibrium']])
# X = np.array(allPopStatistics[['Emean_exhyt','Fix_index','Mlocus_homozegosity_mean','Mlocus_homozegosity_variance','Gametic_disequilibrium']])
# y = np.array(allPopStatistics['Ne'])
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)


# Z_scaled = scaler.transform(Z)


# alphas=np.logspace(1, 12, 13)
# ridge_model = RidgeCV(alphas, cv=5)
# #
# ridge_model.fit(X_train, y_train)
# predict_res = ridge_model.predict(X_test)
# #compare the predicted results with the actual results
# predict_res = np.array(predict_res).astype(float)
#
# # 确保 y_test 也是浮点数类型的数组
# y_test = np.array(y_test).astype(float)
#
# # 现在进行减法操作
# temp = predict_res - y_test
# #绝对值小于10的个数
#
# num_greater = np.sum(np.abs(temp) < 10)
# print("Number of greater: ", num_greater)
# print("Selected alpha: ", ridge_model.alpha_)
# # train_res = ridge_model.predict(X_test)
# # num_greater = np.sum(train_res > y_test)
#
# # print("Number of greater: ", num_greater)
#
# dump(ridge_model, modelName)



