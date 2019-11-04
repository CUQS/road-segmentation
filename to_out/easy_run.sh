#!/bin/bash
python3 run_segmentation.py -m faster_rcnn.om -w 800 -h 600 -i test.png -o ./ -c 21
