#!/bin/bash
mkdir valid_vis
python3 evaluate.py\
            --max-detections 3\
            --score-threshold 0.5\
            --save-path valid_vis\
            --backbone p3d\
            --convert-model $1

#             --gpu 1\
