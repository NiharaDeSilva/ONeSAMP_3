import fwdpy11
import fwdpy11.model_params
import fwdpy11.demography
import numpy as np

# Define simulation parameters
Ne = 1000  # Effective population size
n_loci = 10  # Number of loci
n_individuals = 50  # Number of individuals

# Define the mutation rate and recombination rate
mutation_rate = 1e-8
recombination_rate = 1e-8

# Define demography
demography = fwdpy11.demography.Demography([Ne] * 8)  # Constant population size over generations

# Initialize population
pop = fwdpy11.DiploidPopulation(Ne, n_loci)

# Define model parameters
params = fwdpy11.model_params.SlocusParams(
    nregions=[],
    sregions=[],
    recregions=[fwdpy11.Region(0, n_loci, recombination_rate)],
    rates=(mutation_rate, recombination_rate),
    demography=demography,
)

# Run simulation for 8 generations
for generation in range(8):
    fwdpy11.wright_fisher.Slocus(params, pop)
    if generation >= 2:
        # Extract information from generation 2 to 8
        print(f"Generation {generation + 1}:")
        print(f"Number of individuals: {pop.N}")
        print(f"Genetic diversity: {np.mean(pop.sample_sfs())}")
        # Additional processing can be done here

# Extract final population state
final_state = pop.dump_tables()
print(final_state)
