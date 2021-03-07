#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_PROPOSE GO"
echo "-----"

MODE="CARLA_PROPOSE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_PROPOSE GO DONE"
echo "----------"

