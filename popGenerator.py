from popSimulator import SimulatePopulations
import os

ONESAMP2COAL_MINALLELEFREQUENCY=0.05
mutationRate=0.012
duration_start=2
duration_range=6
missing_data_percentage=0.20
rangeNe=100,500
NeVal=200
numPOP="00256"


outputSampleSizes=(50, 100, 200)
locis=(40, 80, 160, 320)
simulate_populations = SimulatePopulations()

for sampleSize in outputSampleSizes:
    for loci in locis:
        for i in range(1, 11):
            file_name = f"genePop{sampleSize}x{loci}_{i}"
            path = os.path.join("../data/data_V1.3", file_name)
            simulate_populations.generate_input_population(sampleSize, loci, NeVal, mutationRate, path, duration_start, duration_range, missing_data_percentage)

