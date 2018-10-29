#!/bin/bash 
#SBATCH -N 1         
#SBATCH -t 00:30:00  
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jalnliu@lbl.gov
#SBATCH -p debug   
#SBATCH -L SCRATCH  
#SBATCH -C haswell

output="no_tar1"
module load python/3.6-anaconda-5.2
source activate conda_notebook
srun -n 1 python -m cProfile -o $output.log torch_load_bench.py 1024 1 > $output.out
wait
output="no_tar2"
srun -n 1 python -m cProfile -o $output.log torch_load_bench.py 1024 1 > $output.out
wait 
output="no_tar3"
srun -n 1 python -m cProfile -o $output.log torch_load_bench.py 1024 1 > $output.out
