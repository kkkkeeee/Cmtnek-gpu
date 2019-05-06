#!/bin/bash


ARGC=$#
if [ $ARGC -ne 1 ]
then
    echo "Usage: ./coll.timeReinit.sh <input.file> "
    exit 1
fi

#input file
inputFile=$1

grep "GPU compute_entropy time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt > sbtTimeGPU.result.txt
#awk 'END {print NR }' in.txt


grep "GPU entropy_viscosity_gpu comm1 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU entropy_residual_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU entropy_viscosity_gpu double_copy time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU wavevisc_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU resvisc_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU evmsmooth_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU dsavg time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU fluxes_full_field_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU face_state_commo copy1 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU face_state_commo gs_op_fields time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU face_state_commo copy2 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "GPU face_state_commo_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "GPU inviscidbc_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "GPU inviscidflux_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "GPU igu_cmt_gpu_wrapper1 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU igu_cmt cudaMemcpy1 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU igu_cmt gs_op_fields time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "GPU igu_cmt cudaMemcpy2 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt

grep "GPU igu_cmt_gpu_wrapper2 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt

