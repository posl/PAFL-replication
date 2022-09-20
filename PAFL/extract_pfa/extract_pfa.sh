#!/bin/bash

set -e # Abort if one of the commands fail

for dataset in 3 4 7 bp mr imdb toxic mnist; do
for variant in srnn gru lstm; do
  echo dataset=$dataset, variant=$variant
  # original trace取得
  cd ./extract_original_trace 
  echo extract original trace...
  python do_ori_extract.py $dataset $variant --device $1 --use_clean 1

  # abstract trace生成
  cd ../make_abstract_trace
  echo make abstract trace...
  python do_abs_making.py $dataset $variant

  # pfa抽出
  cd ../learning_pfa
  echo extract pfa...
  python do_pfa_extract.py $dataset $variant
  cd ..
done
done