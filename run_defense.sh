#!/bin/bash
#
# run_defense.sh is a script which executes the defense
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_defense.sh INPUT_DIR OUTPUT_FILE
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_FILE - file to store classification labels
#

INPUT_DIR=$1
OUTPUT_FILE=$2

SCRIPT_DIR=$(pwd)
CONV_CONF="${SCRIPT_DIR}/convolutional_confidence_repo/python/"
export PYTHONPATH="$CONV_CONF:${PYTHONPATH}"
ls $CONV_CONF

echo "PYTHONPATH = $PYTHONPATH"

python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_path=checkpoints/model.ckpt-24245
#  --checkpoint_path=checkpoints/model.ckpt-1361
# ens_adv_inception_resnet_v2.ckpt
