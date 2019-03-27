#/bin/bash
step=$1
for (( i = 1; i<=8; i++));
do
    diff res3.id.$(printf %04d $i).step.$step.stage.0001.firstCopy /lustre/atlas/scratch/kekezhai/csc188/GPUCMTnek-012019/uniform_flownogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i is different in res3 before"
    fi
done