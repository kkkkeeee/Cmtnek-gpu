#/bin/bash
step=$1
for (( i = 1; i<=18; i++));
do
for (( j = 1; j<=3; j++));
do
    diff fatface.nqq.$(printf %04d $i).iwp.$(printf %04d $j)step.$step.before /lustre/atlas/scratch/adeesha/csc188/uniform_flownogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "Element $i $j is different in uarray before"
    fi
done
done
