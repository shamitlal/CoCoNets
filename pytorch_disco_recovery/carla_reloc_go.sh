#!/bin/bash

set -e # exit on error

echo "-----"
echo "CARLA_RELOC GO"
echo "-----"

MODE="CARLA_RELOC"
export MODE
python -W ignore main.py

echo "----------"
echo "CARLA_RELOC GO DONE"
echo "----------"

