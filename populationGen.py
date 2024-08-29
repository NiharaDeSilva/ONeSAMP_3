import fwdpy11
import fwdpy11.model_params as mp
import fwdpy11.wright_fisher as wf
import numpy as np

# Set up parameters
pop_size = 100  # Reduced for simplicity
genome_length = 1e6  # Length of the genome in base pairs
num_generations = 8
mutation_rate = 1e-8  # Per base per generation mutation rate
recombination_rate = 1e-8  # Per base per generation recombination rate

# Define a Wright-Fisher simulation
params = mp.ModelParams(
    nregions=[fwdpy11.Region(0, genome_length, 1.0)],
    sregions=[],
    recregions=[fwdpy11.Region(0, genome_length, recombination_rate)],
    rates=(mutation_rate,),
    deme_sizes=[pop_size] * num_generations,
    num_generations=num_generations
)

# Run the simulation
pop = fwdpy11.DiploidPopulation(pop_size, genome_length)
wf.wright_fisher(params, pop)

# Extract SNP data
tables = pop.dump_tables()
positions = tables.sites.position  # SNP positions
genotypes = tables.genotype_matrix()  # Genotype matrix

# Function to convert genotype data to GenePop format
def write_genepop(positions, genotypes, pop_size, output_file):
    num_snps = len(positions)
    with open(output_file, "w") as f:
        # Write header information
        f.write("SLiM Simulation Output\n")  # Title line
        for pos in positions:
            f.write(f"SNP_{int(pos)}\n")  # Loci positions

        # Write population data
        f.write("Pop\n")
        for ind in range(pop_size):
            f.write(f"Ind_{ind + 1}, ")
            for snp in range(num_snps):
                # Each genotype is composed of two alleles for diploid individuals
                allele1 = genotypes[2 * ind, snp]
                allele2 = genotypes[2 * ind + 1, snp]

                # GenePop format requires a 2-digit representation of each allele
                f.write(f"{allele1 + 1:02d}{allele2 + 1:02d} ")
            f.write("\n")

# Write the SNP data to a GenePop file
output_file = "simulation_output.gen"
write_genepop(positions, genotypes, pop_size, output_file)

print(f"GenePop file saved as {output_file}")
