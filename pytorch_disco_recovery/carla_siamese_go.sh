#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_SIAMESE GO"
echo "-----"

MODE="CARLA_SIAMESE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_SIAMESE GO DONE"
echo "----------"

