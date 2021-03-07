#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_SSIAMESE GO"
echo "-----"

MODE="CARLA_SSIAMESE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_SSIAMESE GO DONE"
echo "----------"

