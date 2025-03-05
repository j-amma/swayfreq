#!/bin/bash
#SBATCH --job-name=treesway_frame               # Job name
#SBATCH -p debug                                # Submit to 'debug' partition or queue
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=Joseph.Ammatelli@dri.edu    # Where to send mail.  Set this to your email address
#SBATCH --nodes=1                               # run on 1 node
#SBATCH --ntasks-per-node=1                     # maximum number of tasks per node
#SBATCH --cpus-per-task=4                       # Number of cores per MPI task
#SBATCH --mem-per-cpu=100GB                     # Memory (i.e. RAM) per processor
#SBATCH --comment=Treesway/NewFacultySupport    # Put your PG# or Project name here
#SBATCH --time=00:30:00                         # Wall time limit (days-hrs:min:sec)
#SBATCH --output=treesway_frame.log             # Path to the standard output and error files relative to the working directory

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"

# invoke swayfreq code
conda activate swayfreq
python3 get_freq_whole_frame.py