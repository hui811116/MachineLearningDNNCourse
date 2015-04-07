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
RATE=0.1
BSIZE=256
MAXEPOCH=100
MOMENTUM=0


./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM} \
--inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE} --maxEpoch ${MAXEPOCH}\
--momentum ${MOMENTUM}
