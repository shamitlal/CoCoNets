#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_SOB GO"
echo "-----"

MODE="CARLA_SOB"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_SOB GO DONE"
echo "----------"

