#!/bin/bash
for vari in srnn gru lstm; do
python count_critical_ties.py $vari
done
python make_table.py