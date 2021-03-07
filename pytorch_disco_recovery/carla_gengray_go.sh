#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_GENGRAY GO"
echo "-----"

MODE="CARLA_GENGRAY"
export MODE
python -W ignore main.py
# CUDA_VISIBLE_DEVICES=0 python -W ignore main.py

echo "----------"
echo "CARLA_GENGRAY GO DONE"
echo "----------"

