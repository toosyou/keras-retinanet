#!/bin/bash
python3 predict_dicom.py\
            --max-detections 5\
            --score-threshold 0.05\
            --save-path "$3"\
            --backbone p3d\
            --convert-model $1 "$2"
