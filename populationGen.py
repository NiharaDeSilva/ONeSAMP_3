import fwdpy11 as fp11
import numpy as np

def simulate_snp_data(num_generations, pop_size, num_individuals, num_loci, mutation_rate):
    # Create an initial population of diploid individuals
    population = fp11.DiploidPopulation(pop_size, num_loci)

    # Define the mutation rate and recombination rate (if applicable)
    mut_model = fp11.MutationModel(mutation_rate=mutation_rate)

    # Define the parameters for the simulation
    params = fp11.ModelParams(
        N=pop_size,
        theta=2 * pop_size * mutation_rate * num_loci,  # θ = 4Nμ in diploid model
        rho=0.0,  # Recombination rate (set to 0 for no recombination)
        seed=np.random.randint(1, 100000),
        demography=np.ones(num_generations + 1) * pop_size,  # Constant population size
    )

    # Simulate the population over the specified generations
    rng = fp11.GSLrng(params.seed)
    fwdpy11.wright_fisher(rng, population, params, num_generations)

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
                current_base = alleles[mutation.position]
                possible_mutations = [nuc for nuc in nucleotides if nuc != current_base]
                alleles[mutation.position] = np.random.choice(possible_mutations)
            genotype.append(alleles)
        # Combine alleles into a single sequence for each individual
        combined_sequence = np.array(genotype[0]) + np.array(genotype[1])
        combined_snp_data.append(''.join(combined_sequence))

    return np.array(combined_snp_data)

# Parameters
num_generations = 8
pop_size = 100  # Effective population size
num_individuals = 40  # Number of individuals to sample SNP data from
num_loci = 50  # Number of loci
mutation_rate = 1e-8  # Effective mutation rate per site per generation

# Run simulation
snp_data = simulate_snp_data(num_generations, pop_size, num_individuals, num_loci, mutation_rate)

# Display SNP data
print(snp_data)
