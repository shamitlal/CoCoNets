#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_SUB GO"
echo "-----"

MODE="CARLA_SUB"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_SUB GO DONE"
echo "----------"

