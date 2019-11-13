#!/bin/bash
python3 run_segmentation.py -m model.om -w 224 -h 224 -i test.png -o ./ -c 21
