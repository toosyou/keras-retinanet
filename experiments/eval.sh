#!/bin/bash
mkdir test
python3 evaluate.py\
            --max-detections 10\
            --score-threshold 0.03\
            --save-path test\
            --backbone p3d\
            --convert-model $1

#             --gpu 1\
