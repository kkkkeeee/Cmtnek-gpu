#!/bin/bash


ARGC=$#
if [ $ARGC -ne 1 ]
then
    echo "Usage: ./coll.timeReinit.sh <input.file> "
    exit 1
fi

#input file
inputFile=$1

grep "Compute_primitive_vars_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt > sbtTimeGPU.result.txt
#awk 'END {print NR }' in.txt


grep "GPU Setdtcmt time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Entropy_viscosity_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Compute_transport_props_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Fluxes_full_field_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Surface_integral_full_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Imqqtu_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Imqqtu_dirichlet_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt

grep "Igtu_cmt_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Cmtusrf_gpu_wrapper & compute_gradients time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "Convective_cmt_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt


grep "Viscous_cmt_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "Compute_forcing_gpu_wrapper time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt

grep "GPU Compute_forcing total time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "Igu_cmt_gpu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $3 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



grep "Surface_integral_full_gpu_wrapper second time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeGPU.result.txt



