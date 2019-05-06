#/bin/bash
step=$1
for (( i = 1; i<=18; i++));
do
for (( j = 1; j<=3; j++));
do
    diff fatface.nqq.$(printf %04d $i).iwp.$(printf %04d $j).step.$step.stage.0001.afterfirstintegral /lustre/atlas/scratch/kekezhai/csc188/GPUCMTnek-012019/uniform_flow_1000_nogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i $j is different in uarray before"
    fi
done
done