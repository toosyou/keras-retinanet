#!/bin/bash
./predict.sh working_models/augmented_smallanchor64.h5\
                "/home/toosyou/projects/LungTumor/test_data/dicom/0001/Thorax 1_Lungcare (Adult)/LungCARE 1.0 B40f - 3"\
                predict/0001

./predict.sh working_models/augmented_smallanchor64.h5\
                "/home/toosyou/projects/LungTumor/test_data/dicom/0002/Ct With-Without Contrast-Head And Neck(Skull Base To Clavicle)/CA Vol 1.0 - 9"\
                predict/0002

./predict.sh working_models/augmented_smallanchor64.h5\
                "/home/toosyou/projects/LungTumor/test_data/dicom/0003/Thorax 1_Lungcare (Adult)/LungCARE 1.0 B40f - 3"\
                predict/0003

./predict.sh working_models/augmented_smallanchor64.h5\
                "/home/toosyou/projects/LungTumor/test_data/dicom/0004/Ct Without Contrast-Chest/LungCARE 1.25 - 301"\
                predict/0004

./predict.sh working_models/augmented_smallanchor64.h5\
                "/home/toosyou/projects/LungTumor/test_data/dicom/0005/Ct With-Without Contrast-Chest/LungCARE - 3"\
                predict/0005

./predict.sh working_models/augmented_smallanchor64.h5\
                "/home/toosyou/projects/LungTumor/test_data/dicom/0006/Ct Without Contrast-Chest/LungCARE - 301"\
                predict/0006
