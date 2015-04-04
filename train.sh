DATA=/home/hui/Downloads/MLDS_HW1_RELEASE_v1/
TYPE=fbank/
TRAIN=${DATA}${TYPE}train.ark
TEST=${DATA}${TYPE}test.ark
LABEL=${DATA}label/train.lab
INDIM=69
OUTDIM=48
PHONUM=39
SIZE=1124823
TESTNUM=180406
RATE=0.001
BSIZE=500

./train.app ${TRAIN} ${TEST} ${LABEL} --trainnum ${SIZE} --testnum ${TESTNUM} --labelnum ${SIZE} --outputdim ${OUTDIM}  --inputdim ${INDIM} --phonenum ${PHONUM} --labeldim ${OUTDIM} --rate ${RATE} --batchsize ${BSIZE}
