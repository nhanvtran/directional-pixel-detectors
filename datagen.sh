#!/bin/bash

# Executable for parallel processing of Morris' *.gz files
# Takes as argument the integer in the file name
# For a new dataset, adjust EOSDIR and make sure Morris has given you explicit access

i=$1
EOSDIR=/eos/user/s/swartz/Public/pixelav_ac/phase3_100um_thickness_3x3_y_qxpt_flattened_1_7GeV_Data_Set_6

#xrdfs root://eosuser.cern.ch/ ls $EOSDIR

xrdcp root://eosuser.cern.ch/$EOSDIR/pixel_clusters_d${i}.out.gz pixel_clusters_d${i}.out.gz
gunzip pixel_clusters_d${i}.out.gz

python datagen.py $i
