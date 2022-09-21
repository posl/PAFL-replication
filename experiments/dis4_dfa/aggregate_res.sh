#!/bin/bash
for model in lstm gru srnn; do
for ds in 3 4 7 bp mr imdb toxic; do
python make_res_table.py $model
python aggregate_res.py $model
done
done