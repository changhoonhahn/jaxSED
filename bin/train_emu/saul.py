'''

script for deploying job on perlmutter


'''
import os, sys


def modelb_sed(nsample, seed):
    ''' create slurm script and then submit
    '''
    cntnt = '\n'.join([
        "#!/bin/bash",
        #"#SBATCH --qos=debug",
        #"#SBATCH --time=00:09:59",
        "#SBATCH --qos=regular",
        "#SBATCH --time=00:59:59",
        "#SBATCH --constraint=cpu",
        "#SBATCH -N 1",
        "#SBATCH -J modelb%i" % seed,
        "#SBATCH -o o/modelb%i.o" % seed,
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "source ~/.bashrc",
        "conda activate gqp",
        "",
        "python /global/homes/c/chahah/projects/jaxSED/bin/train_emu/1_seds_modelb.py %i %i" % (nsample, seed),
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


def modelb_pca(iwave):
    ''' create slurm script and then submit
    '''
    cntnt = '\n'.join([
        "#!/bin/bash",
        #"#SBATCH --qos=debug",
        #"#SBATCH --time=00:29:59",
        "#SBATCH --qos=regular",
        "#SBATCH --time=00:59:59",
        "#SBATCH --constraint=cpu",
        "#SBATCH -N 1",
        "#SBATCH -J pca%i" % iwave,
        "#SBATCH -o o/pca%i.o" % iwave,
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "source ~/.bashrc",
        "conda activate gqp",
        "",
        "python /global/homes/c/chahah/projects/jaxSED/bin/train_emu/2_compress_sed.py %i" % iwave,
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


def modelb_emu(iwave, seed): 
    cntnt = '\n'.join([
        "#!/bin/bash",
        "#SBATCH -A desi", 
        #"#SBATCH --qos=debug",
        #"#SBATCH --time=00:29:59",
        "#SBATCH -C gpu",
        "#SBATCH -q shared",
        "#SBATCH -t 05:59:59",
        "#SBATCH -n 1",
        '#SBATCH --gpus-per-task=1', 
        "#SBATCH -J emu%i_%i" % (iwave, seed),
        "#SBATCH -o o/emu%i_%i.o" % (iwave, seed), 
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "module load cudatoolkit/12.2", 
        "module load cudnn/8.9.3_cuda12",
        "module load python", 
        "conda activate jax-gpu",
        "",
        "python /global/homes/c/chahah/projects/jaxSED/bin/train_emu/3_train_emu.py %i" % iwave,
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


# run 2024.04.19
#for i in range(10): 
#    modelb_sed(100000, i)
#modelb_sed(100000, 999) # test SEDs

# run 2024.04.19
#for i in range(6, 10): 
#    modelb_photo(i, 'grzW1W2') 
#modelb_photo(999, 'grzW1W2') 

# run 2024.04.22
#modelb_pca(1)
#modelb_pca(2)
#modelb_pca(3)

# run 2020.04.24
for seed in range(5): 
    modelb_emu(0, seed)
    modelb_emu(1, seed)
    modelb_emu(2, seed)
    modelb_emu(3, seed)
