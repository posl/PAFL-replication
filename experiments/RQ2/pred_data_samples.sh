#!/bin/bash

set -e # Abort if one of the commands fail

datasets=("3" "4" "7" "bp" "mr" "imdb")

for e in ${datasets[@]}; do
  echo dataset=${e}
  python eval_extracted_data.py ${e}
done

python make_pred_result_for_look.py
python aggregate_count_pred_perf.py