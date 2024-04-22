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


# run 2024.04.19
for i in range(10): 
    modelb_sed(100000, i)
modelb_sed(100000, 999) # test SEDs

# run 2024.04.19
#for i in range(6, 10): 
#    modelb_photo(i, 'grzW1W2') 
#modelb_photo(999, 'grzW1W2') 
