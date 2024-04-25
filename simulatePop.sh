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

lociList=(80 320)
sampleSizeList=(50 200)
numReps=100
for loci in "${lociList[@]}"; do
  for sampleSize in "${sampleSizeList[@]}"; do
    for ((i=1; i<=$numReps; i++)); do
      outputFileName="genePop${sampleSize}Ix${loci}L_${i}"
      ONESAMP2COAL_MINALLELEFREQUENCY=0.05
      mutationRate="0.000000012"
      rangeNe=150,250
      theta=0.000048,0.0048
      microsatsOrSNPs=s
      NeVals="00200"
      numPOP="00050"
      ./refactor -t1 -rC -b$NeVals -d1 -u$mutationRate -v${theta} -$microsatsOrSNPs -l$loci -i$sampleSize -o1 -f$ONESAMP2COAL_MINALLELEFREQUENCY -p > /blue/boucher/suhashidesilva/Second/ONeSAMP_3/data/data_V3/$outputFileName

      sleep 1
    done
  done
done
