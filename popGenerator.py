from popSimulator import SimulatePopulations
import os

ONESAMP2COAL_MINALLELEFREQUENCY=0.05
mutationRate=0.012
rangeNe=100,500
theta=0.000048,0.0048
NeVal=200
numPOP="00256"


outputSampleSizes=(50, 100, 200)
locis=(40, 80, 160, 320, 1000, 2000)
simulate_populations = SimulatePopulations()

for sampleSize in outputSampleSizes:
    for loci in locis:
        file_name = f"genePop{sampleSize}x{loci}"
        path = os.path.join("../data_wm/", file_name)
        simulate_populations.generate_input_population(sampleSize, loci, NeVal, mutationRate, path)

