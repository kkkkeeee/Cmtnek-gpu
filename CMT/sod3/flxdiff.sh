#/bin/bash
step=$1
for (( i = 1; i<=32; i++));
do
for ((j = 1; j<=6; j++));
do
    diff flx.f.$(printf %04d $j).e.$(printf %04d $i) /lustre/atlas/scratch/kekezhai/csc188/GPUCMTnek-012019/sod3_nogpu/. >& /dev/null
    if [ $? -ne 0 ]; then
        echo "f $j e $i is different in flx"
    fi
done
done
