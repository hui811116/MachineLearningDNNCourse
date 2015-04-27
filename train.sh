DATA=/home/ahpan/DeepLearningHW1/MachineLearningDNNCourse/Data/MLDS_HW1_RELEASE_v1/
TYPE=mfcc/
TRAIN=${DATA}${TYPE}train.ark
TEST=${DATA}${TYPE}test.ark
LABEL=${DATA}label/train.lab
INDIM=39
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
RATE=0.002
BSIZE=256
MAXEPOCH=20000
MOMENTUM="0 0.3 0.6 0.9"
DECAYSET="1 0.9 0.81 0.72"
DECAY=1
INITMODEL=model/momentExpInit.mdl
MODELDIR=model/initexp
UNIRANGE="0.1 0.5 1 2";
DIM=${INDIM}-128-128-${OUTDIM}


./bin/train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
--momentum 0.9 --outName out.mdl --decay ${DECAY} --range 1.5 --dim ${DIM}

#mkdir -p ${MODELDIR}
#for x in $UNIRANGE
#do
#echo "calling momentum $x ..."
#./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
#--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.9 --load ${INITMODEL} --outName ${MODELDIR}/uni${x}.mdl --decay ${DECAY} --range $x | tee ${MODELDIR}/uni${x}.log
#done

#mkdir -p /model/decexp
#for x in $DECAYSET
#do
#echo "calling momentum $x ..."
#./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
#--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH} \
#--momentum 0.9 --load ${INITMODEL} --outName /model/decexp/dec${x}.mdl --decay $x | tee /model/decexp/dec${x}.log
#done

echo "experiment done!"
