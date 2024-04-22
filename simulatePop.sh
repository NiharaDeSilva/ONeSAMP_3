#!/bin/bash
#SBATCH --job-name=oneSamp    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashi.desilva@gmail.com     # Where to send mail
#SBATCH --ntasks=1		      # Number of tasks
#SBATCH --cpus-per-task=4	      # Number of cores per task
#SBATCH --mem=10gb                     # Job memory request
#SBATCH --time=04:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
exec 2>&1

pwd; hostname; date

module load R/4.1
chmod +rwx /blue/boucher/suhashidesilva/ONeSAMP_3/build/OneSamp

ONESAMP2COAL_MINALLELEFREQUENCY=0.05
mutationRate="0.000000012"
rangeNe=100,500
theta=0.000048,0.0048
microsatsOrSNPs=s
NeVals="00200"
numPOP="00256"

outputSampleSizes=(50 200)
locis=(40 160)

for outputSampleSize in "${outputSampleSizes[@]}"; do
  for loci in "${locis[@]}"; do
    for i in {1..100}; do
      ./refactor -t1 -rC -b$NeVals -d1 -u$mutationRate -v${theta} -$microsatsOrSNPs -l$loci -i$outputSampleSize -o1 -f$ONESAMP2COAL_MINALLELEFREQUENCY -p > "/blue/boucher/suhashidesilva/ONeSAMP_3/data/data_V2/genePop${outputSampleSize}x${loci}_${i}"
    done
  done
done
