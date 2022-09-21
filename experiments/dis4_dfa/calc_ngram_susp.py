import os
import sys 
import csv
import numpy as np
import math
import random
import argparse
from collections import defaultdict
import glob 
import torch
import matplotlib.pyplot as plt
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from utils.constant import START_SYMBOL
from utils.time_util import current_timestamp
from utils.help_func import filter_stop_words, load_pickle, save_pickle, save_json
from utils.constant import *
from model_utils.model_util import load_model
# ゼロ除算を防ぐための定数
EPS = 1e-09

def count_ngram(rows, n, order=True):
  """
  rows中に出現するn-gramの登場回数を数える．

  Parameters
  ------------------
  rows: list of list of str
    各サンプルに対する状態のトレース．
    例えば，
    [
      ['1', '2', '2', '3', '5'],
      ['1', '2', '5'],
      ['1', '2', '4, '7', '3', '4'],
    ]
    みたいな感じ．
  n: int
    n-gramのn．つまり，いくつの状態のまとまりを考えるかを表す数．
  order: bool, default=True
    n-gram内で順番を考慮するかどうか．
    Trueにした場合，例えば(1, 2), (2, 1)は違うbi-gramとして扱うが，Falseだと同じものとして扱う．

  Returns
  ------------------
  ngram_cnt: defaultdict(int)
    n-gram をキーとして，その出現回数を値に持つ辞書．
  """
  ngram_cnt = defaultdict(int)
  for row in rows:
    l = len(row)
    checked = defaultdict(bool)
    for i, s in enumerate(row):
      # i+n > l の場合 n個のかたまりが作れないのでpass
      if (i+n) <= l:
        ngram = row[i:i+n]

        # order=Falseの時はsortしてしまうことで順番の情報をなくす
        if not order:
          ngram.sort()
        
        # そのngramが同じトレース内で既に数えられていた場合は, 重複して数えない
        if not checked[tuple(ngram)]:
            # ngramの出現回数の更新
            ngram_cnt[tuple(ngram)] += 1
            checked[tuple(ngram)] = True
  
  return ngram_cnt

def calc_susp(method, **kwargs):
  """
  スペクトル情報からn-gramごとの疑惑値を返すための関数
  """
  ns, nf, ncs, ncf, nus, nuf = kwargs['ns'], kwargs['nf'], kwargs['ncs'], kwargs['ncf'], kwargs['nus'], kwargs['nuf']
  if method == 'ochiai':
    return ncf / (np.sqrt(nf * (ncf+ncs)) + EPS)
  elif method == 'tarantula':
    return (ncf/(nf+EPS)) / ((ncf/(nf+EPS)) + (ncs/(ns+EPS)) + EPS)
  elif method == 'dstar':
    star = 3
    return (ncf**star) / (ncs + nuf + EPS)
  elif method == 'ochiai2':
    return ncf * nus / (np.sqrt(nf * ns * (ncs+ncf) * (nus+nuf)) + EPS)
  elif method == 'ample':
    return np.abs((ncf/(nf+EPS))-(ncs/(ns+EPS)))
  else:
    raise NotImplementedError(f'FL method {method} is Not Implemented')

if __name__ == '__main__':
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  args = parser.parse_args()
  dataset, model_type = args.dataset, args.model_type
  print(f'----------dataset={dataset}, variant={model_type}----------')
  isTomita = dataset.isdigit()
  method_names = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']

  for i in range(10):
    for k in range(2, 12, 2):
        print(f'----------i={i}, k={k}----------')
        # 成功/失敗時のdfaトレースへのパス
        succ_traces_path = get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}/succ_trace.txt') if not isTomita else \
          get_path(f'data/dfa/{model_type}/tomita_{dataset}/boot_{i}/k={k}/succ_trace.txt')
        fail_traces_path = get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}/fail_trace.txt') if not isTomita else \
          get_path(f'data/dfa/{model_type}/tomita_{dataset}/boot_{i}/k={k}/fail_trace.txt')
        # dfaで予測成功/失敗時のトレースを読み込む
        with open(succ_traces_path, 'r') as f:
          succ_rows = f.readlines()
        with open(fail_traces_path, 'r') as f:
          fail_rows = f.readlines()
        # 各行の末尾の改行を取り除き, コンマで配列を切る
        succ_rows = [row.strip().split(',') for row in succ_rows]
        fail_rows = [row.strip().split(',') for row in fail_rows]
          # 諸々の保存dir
        for n in range(1, 6, 1):
          save_dir = get_path(f'data/dfa/{model_type}/{dataset}/boot_{i}/k={k}/n={n}') if not isTomita else \
            get_path(f'data/dfa/{model_type}/tomita_{dataset}/boot_{i}/k={k}/n={n}')
          # 予測に成功/失敗時のn-gramを数える
          succ_ngram_cnt = count_ngram(succ_rows, n)
          fail_ngram_cnt = count_ngram(fail_rows, n)

          #################################################
          # 成功時,失敗時のngramの出現回数を用いて, 
          # 各ngramに対する疑惑値を計算する
          #################################################
          
          # FLメソッドごとの，ngramに対する疑惑値を保存しておく辞書
          ngram_susp_dict = defaultdict(defaultdict)
          # 予測に成功/失敗した回数
          Ns = len(succ_rows)
          Nf = len(fail_rows)
          # 成功時,失敗時のngramの集合
          ngrams = set( list(succ_ngram_cnt.keys()) + list(fail_ngram_cnt.keys()) )
          for method in method_names:
            for ngram in ngrams:
              # そのngramを通って成功,失敗したサンプル数を保存
              Ncs, Ncf = succ_ngram_cnt[ngram], fail_ngram_cnt[ngram]
              # 通らずに成功/失敗したサンプル数
              Nus, Nuf = Ns-Ncs, Nf-Ncf
              # 各メソッドを使って疑惑値を計算する
              ngram_susp_dict[method][ngram] = calc_susp(method, ns=Ns, nf=Nf, ncs=Ncs, ncf=Ncf, nus=Nus, nuf=Nuf)
            # 疑惑値の降順にソートしておく
            ngram_susp_dict[method] = dict(sorted(ngram_susp_dict[method].items(), key=lambda x:x[1], reverse=True))
            # 計算した疑惑値を所定のディレクトリに保存する
            save_pickle(os.path.join(save_dir, f"{method}_susp_dict.pkl"), ngram_susp_dict[method])
            # 疑惑値の上位20件はすぐみれるようにjsonで保存する
            ngram_susp_dict_top20 = {str(k) : ngram_susp_dict[method][k] for k in list(ngram_susp_dict[method])[:20]}
            save_json(os.path.join(save_dir, f"{method}_susp_top20.json"), ngram_susp_dict_top20)