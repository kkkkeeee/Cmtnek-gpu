#/bin/bash
step=$1
for (( i = 1; i<=63; i++));
do
    diff res3.id.$(printf %04d $i).step.$step.stage.0001.afterUpdateU /lustre/atlas/scratch/kekezhai/csc188/GPUCMTnek-012019/uniform_flow_1000_nogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i is different in uarray before"
    fi
done
