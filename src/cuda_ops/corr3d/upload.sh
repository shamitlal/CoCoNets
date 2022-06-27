#!/bin/bash

rsync -vaP -e "ssh" * matrix.ml.cmu.edu:~/pytorch_correlation_3d/

