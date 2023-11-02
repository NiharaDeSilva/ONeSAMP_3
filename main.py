#!/usr/bin/python
import subprocess
import sys
import argparse

import os
import shutil
import numpy as np
import time
from statistics import statisticsClass
import threading
import concurrent.futures
import warnings
from sklearn.linear_model import LinearRegression

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

##Creating input file with intial statistics
textList = [str(inputFileStatistics.stat1), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
            str(inputFileStatistics.stat4), str(inputFileStatistics.stat5)]
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
    thread_id = threading.get_ident()
    # change the intermediate file name by thread id
    intermediateFilename = str(thread_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
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

# Concurrently process the random populations
with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    with fileALLPOP as result_file:
        for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
            result_file.write('\t'.join(result) + '\n')

fileALLPOP.close()

try:
    shutil.rmtree(path, ignore_errors=True)
except FileExistsError:
    pass

#########################################
# FINISHING ALL POPULATIONS
########################################
# STARTING RSCRIPT
#########################################
# TODO double check there
ALL_POP_STATS_FILE = allPopStats

print("Below are the mean, median, and 95 credible limits ", end="")
print("for the posterior distribution of the effective ")
print("population size from OneSamp\n")

#Ignore all warnings
warnings.filterwarnings("ignore")


def makepd5(target, x, sumstat, tol, gwt, rejmethod=True):
    scaled_sumstat = sumstat.copy()
    for i in range(5):
        scaled_sumstat[:, i] = (sumstat[:, i] - sumstat[gwt, i].mean()) / np.sqrt(sumstat[gwt, i].var())

    target_s = target.copy()
    for i in range(5):
        target_s[i] = (target[i] - sumstat[gwt, i].mean()) / np.sqrt(sumstat[gwt, i].var())

    dist = np.sqrt(np.sum((scaled_sumstat - target_s) ** 2, axis=1))
    dist[~gwt] = np.floor(np.max(dist[gwt]) + 10)

    abstol = np.quantile(dist, tol)
    wt1 = dist < abstol

    if rejmethod:
        l1 = {'x': x[wt1], 'wt': 0}
    else:
        regwt = 1 - (dist[wt1] ** 2) / (abstol ** 2)
        x1 = scaled_sumstat[wt1, 0]
        x2 = scaled_sumstat[wt1, 1]
        x3 = scaled_sumstat[wt1, 2]
        x4 = scaled_sumstat[wt1, 3]
        x5 = scaled_sumstat[wt1, 4]
        fit1 = LinearRegression()
        fit1.fit(np.column_stack((x1, x2, x3, x4, x5)), x[wt1], sample_weight=regwt)
        predmean = fit1.predict(np.array([target_s]))

        fv = fit1.predict(np.column_stack((x1, x2, x3, x4, x5)))

        l1 = {'x': x[wt1] + predmean - fv, 'vals': x[wt1], 'wt': regwt, 'ss': scaled_sumstat[wt1, :], 'predmean': predmean, 'fv': fv}

    return l1



def normalize(x,y):
    mean_y = np.mean(y)
    var_y = np.var(y)
    if not np.isfinite(var_y):
        retval = 0
    elif var_y == 0:
        retval = mean_y
    else:
        retval = (x - mean_y) / np.sqrt(var_y)
    return retval

    

# Get the command-line arguments
# Exclude the first argument, which is the script name
# args = sys.args[1:]

# if len(args)>=2:
#     allfilename = args[0]
#     initialfilename = args[1]
# else:
#     print("Usage: python script.py <allfilename> <initialfilename>")
#     sys.exit(1)


# allfilename = ALL_POP_STATS_FILE
# initialfilename = inputPopStats


initialfilename = inputPopStats    # Replace with the actual file path
numStatistics = 5  # Set the number of statistics

try:
    with open(initialfilename, 'r') as file:
        # Read the data from the file
        data = file.read()
        data = data.split()  # Split the data into individual values

        # Convert the data to a NumPy array and reshape it
        import numpy as np
        m2 = np.array(data, dtype=float)
        m2 = m2.reshape(-1, numStatistics).T  # Transpose the matrix

        # Print the resulting matrix
        print(m2)
except FileNotFoundError:
    print(f"File '{initialfilename}' not found.")

numSamples = m2.shape[0] - 1

mExpected = m2[0,0]
ldExpected = m2[0,1]
lnbExpected = m2[0,2]
hetxExpected = m2[0,3]
xhetExpected = m2[0,4]

# print(mExpected)
# print(ldExpected)
# print(lnbExpected)
# print(hetxExpected)
# print(xhetExpected)


targetStatVals = np.array([mExpected, ldExpected, lnbExpected, hetxExpected, xhetExpected])
# targetStatVals = [mExpected, ldExpected, lnbExpected, hetxExpected, xhetExpected]


standardIn = ALL_POP_STATS_FILE

try:
    with open(allfilename, 'r') as standardIn:
        # Read the data from the file
        data = standardIn.read()
        data = data.split()  # Split the data into individual values

        # Convert the data to a NumPy array and reshape it
        import numpy as np
        m1 = np.array(data, dtype=float)
        m1 = m1.reshape(-1, numStatistics + 1).T  # Transpose the matrix

        # Print the resulting matrix
        print(m1)
except FileNotFoundError:
    print(f"File '{allfilename}' not found.")


numSamples = m1.shape[0] - 1

#Extract column and assign to appropriate statistics
ne = m1[0, :]
m = m1[1, :]
ld = m1[2, :]
lnb = m1[3, :]
hetx = m1[4, :]
xhet = m1[5, :]



# Box Cox transform of Ne data
lambda_value = -0.2

# Calculate neBoxCox using the provided lambda
neBoxCox = ((ne ** lambda_value) - 1) / lambda_value

# Combine columns m, ld, lnb, hetx, xhet into a data matrix
datamatrix = np.columnstack((m,ld,lnb,hetx,xhet))

result1 = makepd5(targetStatVals, neBoxCox, datamatrix, 0.05, np.arange(1, datamatrix.shape[0] + 1), False)

# Inverse Box-Cox transformation
result1['x'] = ((lambda_value * result1['x']) + 1) ** (1 / lambda_value)


# Statistics to compute
# Inverse Box-Cox transform for mean
mean = ((lambda_value * result1['predmean']) + 1) ** (1 / lambda_value)
# Median
median = np.median(result1['x'])  # Using NumPy's median function
# Variance
vari = np.var(result1['x'])  # Using NumPy's var function
# Minimum
min_val = np.min(result1['x'])  # Using NumPy's min function
# Maximum
max_val = np.max(result1['x'])  # Using NumPy's max function


# maybe we should use 0.25 and 0.75?
quantiles = np.percentile(result1['x'], [2.5, 97.5])


# Display the final output
print("min        max        mean        median      lower95CL   upper95CL\n")
print("%.2f      %.2f      %.2f      %.2f      %.2f      %.2f\n", min, max, mean, median, quantiles[0], quantiles[1])


# rScriptCMD = "Rscript %s %s %s" % (FINAL_R_ANALYSIS, ALL_POP_STATS_FILE, inputPopStats)
# print(rScriptCMD)
# res = os.system(rScriptCMD)

# if (res):
#     print("ERROR:main: Could not run Rscript.  FATAL ERROR.")
#     exit()

# if (DEBUG):
#     print("Finish linear regression")

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
