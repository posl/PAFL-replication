#!/bin/sh

set -e # Abort if one of the commands fail

datasets=("3" "4" "7" "bp" "mr" "imdb")
ks=(2 4 6 8 10)
for d in ${datasets[@]}; do
  for boot_id in {0..9}; do
    for k in ${ks[@]}; do
      python make_pfa_trace.py ${d} lstm ${boot_id} $k
    done
  done
done