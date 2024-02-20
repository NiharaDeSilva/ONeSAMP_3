import simuPOP as sim

# # Parameters
# num_generations = 8  # Total number of generations to simulate
# effective_population_size = 100  # Effective population size
# mutation_rate = 0.001  # Mutation rate per locus per generation
# num_loci = 10  # Number of loci per individual
# min_allele_freq = 0.05  # Minimum allele frequency to monitor or manage
#
# # Initialize a diploid population
# pop = sim.Population(size=[effective_population_size], ploidy=2,
#                      loci=[num_loci],
#                      infoFields='alleleFreq')
#
# # Evolution process with mutation and allele frequency tracking
# # simu = sim.Simulator(pop, rep=1)
# pop.evolve(
#     initOps=[
#         sim.InitSex(),
#         sim.InitGenotype(genotype=[0]*20+[1]*20)
#     ],
#     preOps=[
#         sim.Stat(alleleFreq=range(num_loci)),
#         sim.PyOperator(lambda pop: all([freq >= min_allele_freq for freq in pop.dvars().alleleFreq[0].values()])),
#     ],
#     matingScheme=sim.RandomMating(
#         ops=[
#             sim.Recombinator(rates=0.01)
#         ]
#     ),
#     postOps=[
#         sim.SNPMutator(u=mutation_rate, v=mutation_rate),
#         sim.Stat(alleleFreq=range(num_loci), step=1),
#     ],
#     gen=num_generations
# )

def importData(filename):
    'Read data from ``filename`` and create a population'
    data = open(filename)
    header = data.readline()
    fields = header.split(',')
    # columns 1, 3, 5, ..., without trailing '_1'
    names = [fields[x].strip()[:-2] for x in range(1, len(fields), 2)]
    popSize = 0
    alleleNames = set()
    for line in data.readlines():
        # get all allele names
        alleleNames |= set([x.strip() for x in line.split(',')[1:]])
        popSize += 1
# create a population
    alleleNames = list(alleleNames)
    pop = sim.Population(size=popSize, loci=len(names), lociNames=names,
       alleleNames=alleleNames)
# start from beginning of the file again
    data.seek(0)
# discard the first line
    data.readline()
    for ind, line in zip(pop.individuals(), data.readlines()):
        fields = [x.strip() for x in line.split(',')]
        sex = sim.MALE if fields[0] == '1' else sim.FEMALE
        ploidy0 = [alleleNames.index(fields[x]) for x in range(1, len(fields), 2)]
        ploidy1 = [alleleNames.index(fields[x]) for x in range(2, len(fields), 2)]
        ind.setGenotype(ploidy0, 0)
        ind.setGenotype(ploidy1, 1)
        # ind.setSex(sex)
# close the file
    data.close()
    return pop

from simuPOP.utils import saveCSV
pop = sim.Population(size=[50], loci=[40], ploidy=2)
pop.recodeAlleles([0, 0, 1, 3, 2], alleleNames=['A', 'C', 'G', 'T'])
# sim.initSex(pop)
pop.evolve(
    initOps=[
        sim.InitSex(),
        sim.InitGenotype(freq=[0.25, 0.25, 0.25, 0.25])
    ],
     preOps=[
        sim.Stat(effectiveSize=200),
    ],
    matingScheme=sim.RandomMating(),
    gen=6
)
# sim.initGenotype(pop, freq=[0.25, 0.25, 0.25, 0.25])
# output sex but not affection status.
saveCSV(pop, filename='sample.csv', affectionFormatter=None,
            sexFormatter={sim.MALE:1, sim.FEMALE:2})
# have a look at the file
# print(open('sample.csv').read())


pop1 = importData('sample.csv')
sim.dump(pop1)












# pop.recodeAlleles([0, 0, 1, 3, 2], alleleNames=['A', 'C', 'G', 'T'])



# Assuming 'pop' is your population object and the rest of your simulation code is correct
# output_file = "output.gen"  # Specify your output file name

# Corrected method call without named parameters
# pop.save(output_file)
