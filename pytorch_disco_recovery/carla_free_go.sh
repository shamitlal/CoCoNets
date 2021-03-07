#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_FREE GO"
echo "-----"

MODE="CARLA_FREE"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_FREE GO DONE"
echo "----------"