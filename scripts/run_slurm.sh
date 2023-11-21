#!/bin/sh

#SBATCH -c 10              # Request CPU cores
#SBATCH -t 2-00:00          # Runtime in D-HH:MM
#SBATCH -p dl               # Partition to submit to 
#SBATCH --mem=200G           # Request memory
  
~/anaconda3/envs/evalEnv/bin/python synthetic_with_ci_coverage.py  # Command you want to run on the cluster
