#!/bin/bash
for model in lstm gru srnn; do
for ds in 3 4 7 bp mr imdb toxic; do
python calc_ngram_susp.py $ds $model
done
done