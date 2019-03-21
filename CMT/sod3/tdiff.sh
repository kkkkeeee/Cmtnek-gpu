#/bin/bash
step=$1
for (( i = 1; i<=32; i++));
do
    diff t.id.$(printf %04d $i).step.$step.stage.0001.beforeEntropy /lustre/atlas/scratch/kekezhai/csc188/GPUCMTnek-012019/sod3_nogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i is different in uarray before"
    fi
done
