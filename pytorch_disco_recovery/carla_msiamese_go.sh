#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_MSIAMESE GO"
echo "-----"

MODE="CARLA_MSIAMESE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_MSIAMESE GO DONE"
echo "----------"

