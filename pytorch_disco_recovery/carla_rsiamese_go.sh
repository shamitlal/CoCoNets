#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_RSIAMESE GO"
echo "-----"

MODE="CARLA_RSIAMESE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_RSIAMESE GO DONE"
echo "----------"

