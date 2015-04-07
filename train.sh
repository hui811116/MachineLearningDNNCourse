DATA=/home/hui/Downloads/MLDS_HW1_RELEASE_v1/
TYPE=mfcc/
TRAIN=${DATA}${TYPE}train.ark2
TEST=${DATA}${TYPE}test.ark
LABEL=${DATA}label/train.lab2
INDIM=39
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
RATE=0.002
BSIZE=256
MAXEPOCH=20000
MOMENTUM="0 0.3 0.6 0.9"
DECAY=0.99
INITMODEL=model/momentExpInit.mdl
MODELDIR=model/momexp
mkdir -p ${MODELDIR}

#./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
#--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.9 --outName abc.mdl --decay ${DECAY} 
for x in $MOMENTUM
do
echo "calling momentum $x ..."
./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
--momentum $x --load ${INITMODEL} --outName ${MODELDIR}/mom${x}.mdl --decay ${DECAY} | tee ${MODELDIR}/mom${x}.log
done
echo "experiment done!"
