#!/bin/bash

set -e # Abort if one of the commands fail

datasets=("3" "4" "7" "bp" "mr" "imdb")

for e in ${datasets[@]}; do
  echo dataset=${e}, model_type="lstm"

  # original trace取得
  cd ./extract_original_trace
  # echo extract original trace...
  # python do_ori_extract.py ${e} "lstm" 0 0

  # abstract trace生成
  cd ../make_abstract_trace
  echo make abstract trace...
  python do_abs_making.py ${e} "lstm"

  # pfa抽出
  cd ../learning_pfa
  echo extract pfa...
  python do_pfa_extract.py ${e} "lstm" 1
  cd ..
done