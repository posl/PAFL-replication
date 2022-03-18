#!/bin/bash

set -e # Abort if one of the commands fail

datasets=("3" "4" "7" "bp" "mr" "imdb")

for e in ${datasets[@]}; do
  echo dataset=${e}, model_type="lstm"
  python do_pfa_spectrum.py ${e} "lstm"
done