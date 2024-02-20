#!/bin/bash
#SBATCH --job-name=oneSamp    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashi.desilva@gmail.com     # Where to send mail
#SBATCH --ntasks=1		      # Number of tasks
#SBATCH --cpus-per-task=64	      # Number of cores per task
#SBATCH --mem=50gb                     # Job memory request
#SBATCH --time=04:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
exec 2>&1

pwd; hostname; date

module load R/4.1
chmod +rwx /blue/boucher/suhashidesilva/ONeSAMP_3/build/OneSamp

echo "Running plot script on multiple CPU cores"


folder="/blue/boucher/suhashidesilva/ONeSAMP_3/data/datav1"
output="/blue/boucher/suhashidesilva/ONeSAMP_3/output/V8"

#Iterate through the files in the folder
#for file in "$folder"/*; do
#    if [ -f "$file" ]; then
#        filename=$(basename -- "$file")
#        filename_no_extension="${filename%.*}"
#        output_file="$output/${filename_no_extension}"
#        python /blue/boucher/suhashidesilva/ONeSAMP_3/main.py --s 20000 --o "$file" > "$output_file"
#        echo "Processed $file and saved output to $output_file"
#    fi
#done


#python /blue/boucher/suhashidesilva/ONeSAMP_3/main.py --s 20000 --o /blue/boucher/suhashidesilva/ONeSAMP_3/data/datav1/genePop50x40 > /blue/boucher/suhashidesilva/ONeSAMP_3/output/V8/genePop50x40
python /blue/boucher/suhashidesilva/ONeSAMP_3/main.py --s 20000 --o /blue/boucher/suhashidesilva/ONeSAMP_3/data/datav1/genePop50x80 > /blue/boucher/suhashidesilva/ONeSAMP_3/output/V8/genePop50x80
python /blue/boucher/suhashidesilva/ONeSAMP_3/main.py --s 20000 --o /blue/boucher/suhashidesilva/ONeSAMP_3/data/datav1/genePop50x160 > /blue/boucher/suhashidesilva/ONeSAMP_3/output/V8/genePop50x160
python /blue/boucher/suhashidesilva/ONeSAMP_3/main.py --s 20000 --o /blue/boucher/suhashidesilva/ONeSAMP_3/data/datav1/genePop50x320 > /blue/boucher/suhashidesilva/ONeSAMP_3/output/V8/genePop50x320





date
