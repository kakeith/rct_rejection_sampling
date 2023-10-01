 #!/bin/sh

#SBATCH -c 10              # Request CPU cores
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 mins
                            (this requests 2 hours)
#SBATCH -p dl               # Partition to submit to 
                            (should always be dl, for now)
#SBATCH --mem=200G           # Request memory
  
python synthetic_with_ci_coverage.py >> synthetic_out.log 2>&1  # Command you want to run on the cluster
