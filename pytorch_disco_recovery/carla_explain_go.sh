#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_EXPLAIN GO"
echo "-----"

MODE="CARLA_EXPLAIN"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_EXPLAIN GO DONE"
echo "----------"

