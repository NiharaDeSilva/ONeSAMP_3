import fwdpy11
import fwdpy11.ModelParams
import numpy as np

# Actual number of individuals in the population (census population size)
num_individuals = 1000  # Diploid individuals in the population

# Effective population size
Ne = 500  # This is the effective population size

# Number of loci (genome length)
num_loci = 50  # Number of loci in the genome

# Create a population with the actual number of individuals and loci
pop = fwdpy11.DiploidPopulation(num_individuals, num_loci)

# Initialize the random number generator
rng = fwdpy11.GSLrng(54321)

# Define Gaussian Stabilizing Selection with a trait optimum
GSSmo = fwdpy11.GaussianStabilizingSelection.single_trait(
    [
        fwdpy11.Optimum(when=0, optimum=0.0, VS=1.0),  # Initial optimum at 0
        fwdpy11.Optimum(when=10 * Ne - 200, optimum=1.0, VS=1.0),  # Optimum changes
    ]
)

# Recombination rate (Poisson-distributed recombination events)
rho = 1000.

# Define selection on a region with recessive large-effect mutations
des = fwdpy11.GaussianS(beg=0, end=1, weight=1, sd=0.1,
                        h=fwdpy11.LargeEffectExponentiallyRecessive(k=5.0))

# Adjust demography to reflect the effective population size Ne
demography = fwdpy11.ForwardDemesGraph.tubes([Ne], burnin=7)

# Define the parameters for the simulation
p = {
    "nregions": [],  # No neutral regions in this case
    "gvalue": fwdpy11.Additive(2.0),  # Additive genetic values
    "sregions": [des],  # Selected regions
    "recregions": [fwdpy11.PoissonInterval(0, 1.0, rho / float(4 * Ne))],  # Recombination regions using Ne
    "rates": (0.0, 1e-3, None),  # Mutation rates (no neutral mutations, selected mutations)
    "prune_selected": False,  # Do not prune selected mutations
    "demography": demography,  # Demography reflects effective population size Ne
    "simlen": 10 * Ne,  # Simulation length based on effective population size
}

# Create the ModelParams object
params = fwdpy11.ModelParams(**p)

# Run the simulation with simplification interval set to 100 generations
fwdpy11.evolvets(rng, pop, params, simplification_interval=100)

# Extract the SNP/mutation data from the population tables
tree_sequence = pop.dump_tables_to_tskit()

# Output the haplotypes for all individuals in the population
for individual, h in enumerate(tree_sequence.haplotypes()):
    print(f"Individual {individual}: {h}")
