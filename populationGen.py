import fwdpy11 as fp11
import numpy as np

def simulate_snp_data(num_generations, pop_size, num_individuals, num_loci, mutation_rate):
    # Create the population
    population = fp11.DiploidPopulation(pop_size, num_loci)

    scaling = 1.0
    # Create a genetic value function (gvalue) for the simulation
    gvalue = fp11.DiploidGeneticValue(scaling)

    # Create a simulation parameter object
    params = fp11.ModelParams(
        nregions=[fp11.Region(0, num_loci, 1.0)],  # Entire chromosome can mutate
        sregions=[],  # No selection regions
        recregions=[],  # No recombination regions
        rates=(mutation_rate, 0, 0),  # mutation rate, no recombination, no migration
        demography=None,
        gvalue=gvalue
    )

    # Simulate the population over the specified generations
    rng = fp11.GSLrng(np.random.randint(1, 100000))
    fp11.wright_fisher(rng, population, params, num_generations)

    # Sample individuals for SNP data
    sampled_individuals = population.sample(num_individuals)

    # Define nucleotide bases
    nucleotides = ["A", "C", "T", "G"]

    # Convert binary data to A, C, T, G format and combine alleles
    combined_snp_data = []
    for ind in sampled_individuals:
        genotype = []
        for chrom in [ind.a1, ind.a2]:  # diploid has two alleles
            alleles = np.random.choice(nucleotides, num_loci)  # Randomly assign nucleotides
            for mutation in chrom.mutations:
                # Randomly pick a nucleotide different from the current one at the mutation position
                current_base = alleles[mutation.pos]
                possible_mutations = [nuc for nuc in nucleotides if nuc != current_base]
                alleles[mutation.pos] = np.random.choice(possible_mutations)
            genotype.append(alleles)
        # Combine alleles into a single sequence for each individual
        combined_sequence = np.array(genotype[0]) + np.array(genotype[1])
        combined_snp_data.append(''.join(combined_sequence))

    return np.array(combined_snp_data)

# Parameters
num_generations = 8
pop_size = 100  # Effective population size
num_individuals = 10  # Number of individuals to sample SNP data from
num_loci = 1000  # Number of loci
mutation_rate = 1e-8  # Effective mutation rate per site per generation

# Run simulation
snp_data = simulate_snp_data(num_generations, pop_size, num_individuals, num_loci, mutation_rate)

# Display SNP data
print(snp_data)
