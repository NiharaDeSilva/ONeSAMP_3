#!/usr/bin/python
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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge, RidgeCV, LassoCV
from joblib import dump


NUMBER_OF_STATISTICS = 5
t = 30
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
OUTPUTFILENAME = "priors.txt"

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
directory = "temp"
path = os.path.join("./", directory)

POPULATION_GENERATOR = "./build/OneSamp"
FINAL_R_ANALYSIS = "./scripts/rScript.r"

folder_path = 'data/genePop50x40/'
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
t = time.time()
# Loop over each input file to process statistics
file_names = os.listdir(folder_path)
inputList = []
for file_name in file_names:
    fileName = os.path.join(folder_path, file_name)
    inputFileStatistics = statisticsClass()

    # Read and process data
    inputFileStatistics.readData(fileName)
    inputFileStatistics.filterIndividuals(indivMissing)
    inputFileStatistics.filterLoci(lociMissing)
    if (args.n):
        inputFileStatistics.filterMonomorphicLoci()
    inputFileStatistics.test_stat1()
    inputFileStatistics.test_stat2()
    inputFileStatistics.test_stat3()
    inputFileStatistics.test_stat4()
    inputFileStatistics.test_stat5()

    # Create list of statistics for output
    textList = [
        str(inputFileStatistics.stat1),
        str(inputFileStatistics.stat2),
        str(inputFileStatistics.stat3),
        str(inputFileStatistics.stat4),
        str(inputFileStatistics.stat5)
    ]
    inputList.append(textList)  # Append the statistics list to inputList


# File for all input population stats
inputPopStats = "inputPopStats_" + getName(file_name.split('_')[0]) + ".tsv"
with open(inputPopStats, 'w+') as result_file:
    for result in inputList:
        result_file.write('\t'.join(result) + '\n')

# Load data for predictions (assuming each row is a separate prediction point)
input_df = pd.read_csv(inputPopStats, sep='\t')

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

simulate_populations = SimulatePopulations()

# File for all population stats
allPopStats = "allPopStats_" + getName(fileName) + "_" + str(t)
fileALLPOP = open(allPopStats, 'w+')

# Generate random populations and calculate summary statistics
def processRandomPopulation(x):
    loci = inputFileStatistics.numLoci
    sampleSize = inputFileStatistics.sampleSize
    proc = multiprocessing.Process()
    process_id = os.getpid()
    # change the intermediate file name by process id
    intermediateFilename = str(process_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
    intermediateFile = os.path.join(path, intermediateFilename)
    cmd = "%s -u%.9f -v%s -rC -l%d -i%d -d%s -s -t1 -b%s -f%f -o1 -p > %s" % (POPULATION_GENERATOR, mutationRate, rangeTheta, loci, sampleSize, rangeDuration, rangeNe, minAlleleFreq, intermediateFile)
    #simulate_populations.generate_population_data(sampleSize, loci, rangeNe, mutationRate, intermediateFile, duration_start, duration_range, missing_data_percentage)
   
    if (DEBUG):
        print(cmd)

    returned_value = os.system(cmd)

    if returned_value:
        print("ERROR:main:Refactor did not run")

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
    textList = []
    textList = [str(refactorFileStatistics.NE_VALUE),
                str(refactorFileStatistics.stat1),str(inputFileStatistics.stat2),
                str(refactorFileStatistics.stat3),
                str(refactorFileStatistics.stat4),
                str(refactorFileStatistics.stat5)]
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


# Write all population stats to a file to pass as input for Rscript
with fileALLPOP as result_file:
    for result in results_list:
        result_file.write('\t'.join(result) + '\n')

ALL_POP_STATS_FILE = allPopStats

try:
   shutil.rmtree(path, ignore_errors=True)
except FileExistsError:
   pass
fileALLPOP.close()



#########################################
# FINISHING ALL POPULATIONS
########################################
# STARTING LINEAR REGRESSION
#########################################
'''

rScriptCMD = "Rscript %s %s %s" % (FINAL_R_ANALYSIS, ALL_POP_STATS_FILE, inputPopStats)
print(rScriptCMD)
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
allPopStatistics = pd.DataFrame(results_list, columns=['Ne', 'Emean_exhyt','Fix_index','Mlocus_homozegosity_mean','Mlocus_homozegosity_variance','Gametic_disequilibrium'])
# inputStatsList = pd.DataFrame([inputStatsList], columns=['Emean_exhyt','Fix_index','Mlocus_homozegosity_mean','Mlocus_homozegosity_variance','Gametic_disequilibrium'])
X_train = np.array(allPopStatistics[['Emean_exhyt','Fix_index','Mlocus_homozegosity_mean','Mlocus_homozegosity_variance','Gametic_disequilibrium']])
y = np.array(allPopStatistics['Ne'])

y_train = np.array([float(value) for value in y if float(value) > 0])
# X_train, _, y_train, _ = train_test_split(X, y, test_size=0.0, random_state=40)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Create Ridge regression model with cross-validation over alpha values
ridge_model = RidgeCV(alphas=np.logspace(0, 12, 13), cv=5)
ridge_model.fit(X_scaled, y_train)
dump(ridge_model, 'ridge_200.joblib')
#num_greater = np.sum(y_pred > y_test)
#print(num_greater)

'''
# Fit the linear regression model
#dump(result, ‘linear_regression.joblib’)
#y_pred = result.predict(X_test)

# prediction_ridge = ridge_model.predict(Z_scaled)
# print("Selected alpha: ", ridge_model.alpha_)
# print("regularized regression: ", prediction_ridge)
# 
# n_bootstraps = 1000
# # Array to store the predictions for each bootstrap sample
# predictions_samples = np.zeros((n_bootstraps, Z_scaled.shape[0]))
# 
# # Perform bootstrap for prediction intervals
# for i in range(n_bootstraps):
#     # Bootstrap sample indices
#     indices = resample(np.arange(Z_scaled.shape[0]))
#     # Make predictions on the bootstrap sample
#     predictions_samples[i] = ridge_model.predict(Z_scaled[indices])
# 
# # Calculate 95% prediction intervals for each prediction
# prediction_intervals = np.percentile(predictions_samples, [2.5, 97.5], axis=0)
# 
# # Display prediction intervals for each prediction
# print("95% prediction intervals for each prediction:")
# print(prediction_intervals)

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

'''
print("----- %s seconds -----" % (time.time() - start_time))


