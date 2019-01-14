#/bin/bash
step=$1
for (( i = 1; i<=8; i++));
do
    diff vdiff.id.$(printf %04d $i).step.$step.before /lustre/atlas/scratch/adeesha/csc188/sod3nogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i is different in uarray before"
    fi
done
