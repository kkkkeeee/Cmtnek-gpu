#!/bin/bash


ARGC=$#
if [ $ARGC -ne 1 ]
then
    echo "Usage: ./coll.timeReinit.sh <input.file> "
    exit 1
fi

#input file
inputFile=$1

grep "CPU Compute_primitive_vars time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt > sbtTimeCPU.result.txt
#awk 'END {print NR }' in.txt


grep "CPU Setdtcmt time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU Entropy_viscosity time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU Compute_transport_props time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU Fluxes_full_field time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU Surface_integral_full time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU Imqqtu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU Imqqtu_dirichlet time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt

grep "CPU Igtu_cmt time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt



grep "CPU Compute_forcing total time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt



grep "CPU Igu_cmt time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt



grep "CPU Surface_integral_full second time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt



