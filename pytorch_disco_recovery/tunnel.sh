#!/bin/bash

# e.g.:
# ./tunnel.sh 3456 0-36

PORT_NUM=${1}
NODE=${2}

HOST=`hostname`
echo "About to open a tunnel..."
if [[ "$HOST" == *"matrix.ml.cmu.edu"* ]]; then
    echo "OK good, we are on the head node..."
    echo "setting PORT_NUM=${PORT_NUM}"
    echo "setting NODE=${NODE}"
    echo "opening tunnel..."
    ssh -N -4 -L :${PORT_NUM}:localhost:${PORT_NUM} compute-${NODE}
else
    echo "No, it seems we are not on the head node."
fi
