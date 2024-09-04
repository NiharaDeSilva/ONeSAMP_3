import fwdpy11
import fwdpy11.ModelParams
import numpy as np

pop = fwdpy11.DiploidPopulation(500, 50)

rng = fwdpy11.GSLrng(54321)

GSSmo = fwdpy11.GaussianStabilizingSelection.single_trait(
    [
        fwdpy11.Optimum(when=0, optimum=0.0, VS=1.0),
        fwdpy11.Optimum(when=10 * pop.N - 200, optimum=1.0, VS=1.0),
    ]
)

rho = 1000.

des = fwdpy11.GaussianS(beg=0, end=1, weight=1, sd=0.1,
    h=fwdpy11.LargeEffectExponentiallyRecessive(k=5.0))

p = {
    "nregions": [],
    "gvalue": fwdpy11.Additive(2.0, GSSmo),
    "sregions": [des],
    "recregions": [fwdpy11.PoissonInterval(0, 1., rho / float(4 * pop.N))],
    "rates": (0.0, 1e-3, None),
    "prune_selected": False,
    "demography": fwdpy11.ForwardDemesGraph.tubes([pop.N], burnin=10),
    "simlen": 10 * pop.N,
}
params = fwdpy11.ModelParams(**p)

fwdpy11.evolvets(rng, pop, params, simplification_interval=100)

# Extract the SNP/mutation data from the population tables
tables = pop.dump_tables()

# Print SNP data: mutation position, derived state, and frequency
print("SNP Data:")
for mutation in tables.mutations:
    print(f"Mutation ID: {mutation.id}, Position: {mutation.position}, Derived State: {mutation.derived_state}, Frequency: {mutation.freq}")

# Print individuals' genotypes (which mutations they carry)
print("\nIndividuals' Genotypes:")
for ind in tables.individuals:
    print(f"Individual {ind.id} Genotypes: {ind.metadata}")

# Print the mutation positions and alleles carried by individuals
print("\nGenotypes by individual:")
for h in tables.haplotypes:
    print(f"Individual {h.individual} carries mutation {h.mutation}")

