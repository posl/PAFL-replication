#!/bin/bash

set -e # Abort if one of the commands fail

# extract data
for dataset in 3 4 7 bp mr imdb toxic mnist; do
for variant in srnn gru lstm; do
  echo dataset=$dataset, variant=$variant
  python do_data_extract.py $dataset $variant
  python eval_extracted_data.py $dataset $variant
done
done

# evaluate of ex. data
for variant in srnn gru lstm; do
  python make_pred_result_for_look.py $variant
  python average_pred_result_for_look.py $variant
done