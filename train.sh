#!/bin/bash
#SBATCH -o /proj/tinyml_htg_ltu/users/x_floim/logs/digits/%j.out
#SBATCH -e /proj/tinyml_htg_ltu/users/x_floim/logs/digits/%j.err
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 4                           # one CPU core
#SBATCH -t 3-00:00:00
#SBATCH --mem=40G


# conda init bash
source /home/x_floim/miniconda3/etc/profile.d/conda.sh
conda activate htr

# Parameters
main_script=/proj/tinyml_htg_ltu/users/x_floim/Git/DRANet-pytorch/train.py


echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE
PATHLOG="/proj/tinyml_htg_ltu/users/x_floim/logs/digits/${CURRENTDATE}_ID_${SLURM_JOB_ID}/"
echo "path log :"
echo ${PATHLOG}
mkdir ${PATHLOG}

output_file="${PATHLOG}/${SLURM_JOB_ID}.txt"

export PYTHONPATH=/proj/tinyml_htg_ltu/users/x_floim/Git/DeepJDOT_pytorch/


# The job
# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. Cette option n'a pas d'effet sur le flux d'entrée standard
python -u $main_script -T clf -D M MM --ex M2MM
>> $output_file


