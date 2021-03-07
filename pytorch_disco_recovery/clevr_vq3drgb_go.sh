#!/bin/bash

set -e # exit on error

echo "-----"
echo "CLEVR_VQ3DRGB GO"
echo "-----"

MODE="CLEVR_VQ3DRGB"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CLEVR_VQ3DRGB GO DONE"
echo "----------"

