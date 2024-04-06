#!/bin/bash
#SBATCH --job-name=ood_sweep
#SBATCH --account=kempner_barak_lab
#SBATCH --output=/n/holyscratch01/barak_lab/Users/rosieyzh/ood_random_var/logs/%A_%a.log
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1     # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=200GB		# All memory on the node
#SBATCH --partition=kempner
#SBATCH --array=1-25%5

source activate olmo

cd /n/holystore01/LABS/barak_lab/Users/rosieyzh/transformers_ssm_copy/synthetic_exps

srun --jobid $SLURM_JOBID bash -c "python3 run_sweep.py --task_id=${SLURM_ARRAY_TASK_ID} --job_id=${SLURM_ARRAY_JOB_ID}  --num_inits=5 --num_data=5"