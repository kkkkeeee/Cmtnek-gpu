#!/bin/bash


ARGC=$#
if [ $ARGC -ne 1 ]
then
    echo "Usage: ./coll.timeReinit.sh <input.file> "
    exit 1
fi

#input file
inputFile=$1

grep "CPU compute_entropy time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt > sbtTimeCPU.result.txt
#awk 'END {print NR }' in.txt


grep "CPU entropy_viscosity comm1 time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU entropy_residual time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU entropy_viscosity double_copy time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU wavevisc time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU resvisc time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU evmsmooth time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU dsavg time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU fillq+faceu time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU face_state_commo gs_op time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt

grep "CPU face_state_commo time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt



grep "CPU inviscidbc time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt



grep "CPU inviscidflux time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt



grep "CPU igu_cmt_cp+cmult time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $4 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU igu_cmt gs_op_fields time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt


grep "CPU igu_cmt last time" $1 | tee in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
sed -i '1d' in.txt
awk '{ total += $5 } END { print total/NR }' in.txt >> sbtTimeCPU.result.txt

