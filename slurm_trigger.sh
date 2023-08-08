#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu= 100M
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hmgetafix@gmail.com

# Load the necessary modules
module load anaconda3/5.2.0
eval "$(conda shell.bash hook)"
conda activate lag_conda

# Run your Python script with $SLURM_ARRAY_TASK_ID as argument
python example_slurmjob.py