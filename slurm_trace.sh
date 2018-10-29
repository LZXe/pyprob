#!/bin/bash 
#SBATCH -N 1         
#SBATCH -t 01:30:00  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalnliu@lbl.gov
#SBATCH -p regular   
#SBATCH -L SCRATCH  
#SBATCH -C haswell

module load python/3.6-anaconda-5.2
source activate conda_notebook
for i in 0 1 2 3 4 5 6 7 8 9 
do
 srun -n 1 python trace_analysis.py 1244 $i
done

