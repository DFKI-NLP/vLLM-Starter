#!/bin/bash
#SBATCH --job-name=rolm-query
#SBATCH --partition=A100-PCI
#SBATCH --nodelist=serv-3316
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --output=rolm_query.out
#SBATCH --error=rolm_query.err

# DIRECTLY use the python binary from your conda env
/netscratch/<USERNAME>/miniconda3/envs/<VIRTUAL_ENVIRONMENT_NAME>/bin/python <SCRIPT_NAME>
