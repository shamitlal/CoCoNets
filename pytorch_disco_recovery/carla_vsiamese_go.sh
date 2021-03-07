#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_VSIAMESE GO"
echo "-----"

MODE="CARLA_VSIAMESE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_VSIAMESE GO DONE"
echo "----------"

