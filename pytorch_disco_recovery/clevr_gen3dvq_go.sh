#!/bin/bash

set -e # exit on error

echo "-----"
echo "CLEVR_GEN3DVQ GO"
echo "-----"

MODE="CLEVR_GEN3DVQ"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CLEVR_GEN3DVQ GO DONE"
echo "----------"

