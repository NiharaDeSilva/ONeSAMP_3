import fwdpy11 as fp11
import numpy as np
import demes

def simulate_snp_data(num_generations, pop_size, num_individuals, num_loci, mutation_rate):
    # Create the demographic model using the `demes` library
    yaml="""
    description: An example demes model
    time_units: generations
    demes:
     - name: ancestor1
       epochs:
        - start_size: 100
          end_time: 50
     - name: ancestor2
       epochs:
        - start_size: 250
          end_time: 50
     - name: admixed
       start_time: 50
       ancestors: [ancestor1, ancestor2]
       proportions: [0.90, 0.10]
       epochs:
        - start_size: 100
    """

    # Convert the demes graph to a ForwardDemesGraph for fwdpy11
    demography = fp11.ForwardDemesGraph.from_demes(yaml, burnin=1000, burnin_is_exact=False, round_non_integer_sizes=True)

    # Create the population
    population = fp11.DiploidPopulation(pop_size, num_loci)

    # Define the regions where mutations can occur
    nregions = [fp11.Region(0, num_loci, 1.0)]  # Entire chromosome can mutate

    # Create a genetic value function (gvalue) with a scaling parameter
    gvalue = fp11.Additive(1.0)  # Using an additive genetic value model with default scaling

    # Simulation parameter object
    params = fp11.ModelParams(
        nregions=nregions,  # Mutation regions
        sregions=[],  # No selection regions
        recregions=[],  # No recombination regions
        rates=(mutation_rate, 0.0, 0.0),  # (mutation rate, recombination rate, migration rate)
        gvalue=gvalue,  # Specify the genetic value function
        demography=demography  # Specify the ForwardDemesGraph object
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
num_generations = 8  # Number of generations
pop_size = 100  # Effective population size
num_individuals = 10  # Number of individuals to sample SNP data from
num_loci = 1000  # Number of loci
mutation_rate = 1e-8  # Mutation rate per site per generation

# Run simulation
snp_data = simulate_snp_data(num_generations, pop_size, num_individuals, num_loci, mutation_rate)

# Display SNP data
print(snp_data)
