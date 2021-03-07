#!/bin/bash

set -e # exit on error

echo "-----"
echo "INTPHYS_TEST GO"
echo "-----"

MODE="INTPHYS_TEST"
export MODE
python -W ignore main.py

echo "----------"
echo "INTPHYS_TEST GO DONE"
echo "----------"

