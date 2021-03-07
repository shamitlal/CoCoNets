#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_CSIAMESE GO"
echo "-----"

MODE="CARLA_CSIAMESE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_CSIAMESE GO DONE"
echo "----------"

