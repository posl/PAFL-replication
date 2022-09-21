#!/bin/sh

set -e # Abort if one of the commands fail

for dataset in 3 4 7 bp mr imdb toxic mnist; do
  for variant in srnn gru lstm; do
    for i in {0..9}; do
      for k in 2 4 6 8 10; do
        python make_pfa_trace.py $dataset $variant $i $k
      done
    done
  done
done