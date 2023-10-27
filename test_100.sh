#!/bin/bash
#SBATCH --job-name=oneSamp    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashi.desilva@gmail.com     # Where to send mail
#SBATCH --ntasks=5                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
pwd; hostname; date

module load R/4.1
chmod +rwx /blue/boucher/suhashidesilva/ONeSAMP_3/build/OneSamp

echo "Running plot script on multiple CPU cores"

python /blue/boucher/suhashidesilva/ONeSAMP_3/main.py --s 20000 --o /blue/boucher/suhashidesilva/ONeSAMP_3/DataSet/genePop100Ix100L > /blue/boucher/suhashidesilva/ONeSAMP_3/genePop100Ix100L.out

date
