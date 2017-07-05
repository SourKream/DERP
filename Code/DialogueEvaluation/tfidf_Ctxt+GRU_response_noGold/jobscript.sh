#!/bin/bash
#PBS -N tfidfctxt_grupresp_snair
#PBS -P cse
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=24:00:00

## SPECIFY JOB NOW
JOBNAME=nogold_tfidfctxt_gruresp_snair
CURTIME=$(date +%Y%m%d%H%M%S)
cd $PBS_O_WORKDIR

module load mkl/intel/psxe2015/mklvars
module load lib/hdf/5/1.8.16/gnu
module load compiler/cuda/7.0/compilervars
module load suite/intel/parallelStudio
module load lib/caffedeps/master/intel
module load lib/hdf/4/4.2.11/gnu
module load compiler/python/2.7.10/compilervars
CODE_DIR=/home/cse/dual/cs5130275/DERP/Code/Models/tfidf_Ctxt+GRU_response
LOG_DIR=/home/cse/dual/cs5130275/DERP/Code/Models/tfidf_Ctxt+GRU_response/logs
ERR_DIR=/home/cse/dual/cs5130275/DERP/Code/Models/tfidf_Ctxt+GRU_response/logs
/home/cse/dual/cs5130275/anaconda/bin/python $CODE_DIR/tfidfctxt_gruresp.py > $LOG_DIR/log_bs512.txt 2> $ERR_DIR/err_bs512.txt
