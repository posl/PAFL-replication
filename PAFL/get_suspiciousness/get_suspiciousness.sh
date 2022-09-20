#!/bin/bash

set -e # Abort if one of the commands fail

for variant in srnn gru lstm; do
  echo variant=$variant
  python do_calc_ngram_susp.py $variant 1 2 3 4 5
done