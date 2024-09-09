import fwdpy11
import fwdpy11.ModelParams
import numpy as np

# Effective population size
Ne = 200  # This is the effective population size

# Number of loci (genome length)
num_loci = 50  # Number of loci in the genome

# Number of individuals
num_individuals = 100  # Diploid individuals in the population

# Create a population with the desired number of individuals and loci
pop = fwdpy11.DiploidPopulation(num_individuals, num_loci)

# Initialize the random number generator
rng = fwdpy11.GSLrng(54321)

# Define Gaussian Stabilizing Selection with a trait optimum
GSSmo = fwdpy11.GaussianStabilizingSelection.single_trait(
    [
        fwdpy11.Optimum(when=0, optimum=0.0, VS=1.0),  # Initial optimum at 0
        fwdpy11.Optimum(when=10 * pop.N - 200, optimum=1.0, VS=1.0),  # Optimum changes
    ]
)

# Recombination rate (Poisson-distributed recombination events)
rho = 1000.

# Define selection on a region with recessive large-effect mutations
des = fwdpy11.GaussianS(beg=0, end=1, weight=1, sd=0.1,
                        h=fwdpy11.LargeEffectExponentiallyRecessive(k=5.0))

# Define the parameters for the simulation
p = {
    "nregions": [],  # No neutral regions in this case
    "gvalue": fwdpy11.Additive(2.0),  # Additive genetic values
    "sregions": [des],  # Selected regions
    "recregions": [fwdpy11.PoissonInterval(0, 1.0, rho / float(4 * pop.N))],  # Recombination regions
    "rates": (0.0, 1e-3, None),  # Mutation rates (no neutral mutations, selected mutations)
    "prune_selected": False,  # Do not prune selected mutations
    "demography": fwdpy11.ForwardDemesGraph.tubes([Ne], burnin=7),  # Set Ne and burnin for 8 generations
    "simlen": 10 * pop.N,  # Length of the simulation (10x population size)
}

# Create the ModelParams object
params = fwdpy11.ModelParams(**p)

# Run the simulation with simplification interval set to 100 generations
fwdpy11.evolvets(rng, pop, params, simplification_interval=100)

# Extract the SNP/mutation data from the population tables
tree_sequence = pop.dump_tables_to_tskit()

# Prepare and format the output of diploid genotypes by individual
def format_genotypes(tree_sequence, num_individuals):
    # Get haplotypes for both ploidies
    haplotypes = list(tree_sequence.haplotypes())

    for individual in range(0, num_individuals, 2):  # Step by 2 to pair diploids
        genotype_1 = haplotypes[individual]    # Ploidy 1
        genotype_2 = haplotypes[individual + 1]  # Ploidy 2

        # Format the output to display each individual's two ploidies together
        combined_genotype = ''.join([g1 + g2 for g1, g2 in zip(genotype_1, genotype_2)])

        print(f"ind {individual//2 + 1}: {combined_genotype}")

# Output the formatted genotypes
format_genotypes(tree_sequence, num_individuals)
