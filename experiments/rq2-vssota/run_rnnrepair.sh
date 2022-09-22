#!/bin/bash
for ds in 3 4 7 bp mr imdb toxic mnist; do
for vari in srnn gru lstm; do
python make_con_trace.py $ds $vari
done
done

for ds in 3 4 7 bp mr imdb toxic mnist; do
for vari in srnn gru lstm; do
for theta in 10 50; do
python rnnrepair_core.py $ds $vari $theta
done
done
done

for vari in srnn gru lstm; do
python aggregate_sota_result.py $vari
done

for vari in srnn gru lstm; do
for theta in 10 50; do
python aggregate_pafl_result.py $vari $theta
done
done