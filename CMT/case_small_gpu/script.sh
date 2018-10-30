#!/bin/sh

#MSUB -q pbatch
#MSUB -l nodes=25
#MSUB -l partition=vulcan
#MSUB -l walltime=3:00:00
#MSUB -V
#MSUB -l gres=lscratchv

rm -f partxyz*
rm -f partdata*
rm -f blast2d0.f*
echo blast2d > SESSION.NAME
echo `pwd`'/' >> SESSION.NAME
srun -N 25 -n 388 ./nek5000 > output.txt.072018
