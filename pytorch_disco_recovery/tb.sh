#!/bin/bash

# e.g.:
# ./tb.sh 3456 logs_carla_sta

PORT_NUM=${1}
LOG_DIR=${2}

HOST=`hostname`
echo "About to start tensorboard..."
# if [[ "$HOST" == *"compute"* ]]; then
echo "OK good, we are on a compute node..."
echo "setting PORT_NUM=${PORT_NUM}"
echo "setting LOG_DIR=${LOG_DIR}"
echo "starting tensorboard..."
export TMPDIR=/tmp/$USER; mkdir -p $TMPDIR; tensorboard --port=${PORT_NUM} --logdir=${LOG_DIR} --host localhost
# else
#     echo "No, it seems we are not on a compute node."
# fi
