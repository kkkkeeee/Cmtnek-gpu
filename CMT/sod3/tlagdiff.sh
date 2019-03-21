#/bin/bash
step=$1
for (( i = 1; i<=32; i++));
do
    diff tlag.id.$(printf %04d $i).step.$step.stage.0001.afterEntropy /lustre/atlas/scratch/kekezhai/csc188/GPUCMTnek-012019/sod3_nogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i is different in res2 before"
    fi
done
