#!/bin/bash

set -e # Abort if one of the commands fail

for dataset in 3 4 7 bp mr imdb toxic mnist; do
for variant in srnn gru lstm; do
  echo dataset=$dataset, variant=$variant
  python do_pfa_spectrum.py $dataset $variant
done
done