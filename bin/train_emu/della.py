''' script for deploying job on della 
'''
import os, sys


def train_nn(iwave):
    ''' create slurm script and then submit
    '''
    cntnt = '\n'.join([ 
        "#!/bin/bash",
        "#SBATCH --job-name=train_mlp",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem-per-cpu=4G", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --time=04:00:00",
        "$SBATCH -o o/train_mlp%i.o" % iwave, 
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "", 
        "module purge", 
        "module load anaconda3/2024.2", 
        "conda activate jax-gpu", 
        "", 
        "python /home/chhahn/projects/jaxSED/bin/train_emu/3_train_emu.py %i" % iwave, 
        "",
        'now=$(date +"%T")',
        'echo "end time ... $now"',
        ""])

    # create the slurm script execute it and remove it
    f = open('_train.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _train.slurm')
    os.system('rm _train.slurm')
    return None

