#/bin/bash
step=$1
for (( i = 1; i<=5; i++));
do
for (( j = 1; j<=3; j++));
do
    diff graduf.eq.$(printf %04d $i).iwp.$(printf %04d $j).step.$step.stage.0001.afterIgtu /lustre/atlas/scratch/kekezhai/csc188/GPUCMTnek-012019/sod3_nogpu/. >& /dev/null
    #diff fatface.nqq.$(printf %04d $i).iwp.$(printf %04d $j).step.$step.stage.0001.afterImqqtud fatface.nqq.$(printf %04d $i).iwp.$(printf %04d $j).step.$step.stage.0001.afterImqqtu >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i $j is different in uarray before"
    fi
done
done
