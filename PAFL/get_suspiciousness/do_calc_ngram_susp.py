import os
import sys
import numpy as np
import time
from pprint import pprint
import argparse
from collections import defaultdict
#同じ階層のディレクトリからインポートするために必要
sys.path.append("../")
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
# 異なる階層utilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, save_json, load_json
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
    return (ncf/nf) / ((ncf/nf) + (ncs/ns) + EPS)
  elif method == 'dstar':
    star = 3
    return (ncf**star) / (ncs + nuf + EPS)
  elif method == 'ochiai2':
    return ncf * nus / (np.sqrt(nf * ns * (ncs+ncf) * (nus+nuf)) + EPS)
  elif method == 'ample':
    return np.abs((ncf/nf)-(ncs/ns))
  else:
    raise NotImplementedError(f'FL method {method} is Not Implemented')

if __name__=="__main__":
  # bootstrap のサンプル数
  B = 10

  parser = argparse.ArgumentParser()
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("nlist", type=int, nargs='+', help='the list of n of n-gram')
  parser.add_argument("--start_boot_id", type=int, help='What boo_id starts from.', default=0)
  parser.add_argument("--start_k", type=int, help='What k starts from.', default=2)
  args = parser.parse_args()
  model_type, n_list = args.model_type, args.nlist
  start_boot_id, start_k = args.start_boot_id, args.start_k

  datasets = ['tomita_3', 'tomita_4', 'tomita_7',\
              DataSet.BP, DataSet.MR, DataSet.IMDB, DataSet.MNIST, DataSet.TOXIC]
  method_names = ['ochiai', 'tarantula', 'dstar', 'ochiai2', 'ample']

  elapsed_times = np.zeros((B, len(datasets), 5, 5))

  # pfa上での各データによる状態遷移の読み込み
  for i in range(start_boot_id, B):
    print("========= use bootstrap sample {} for training data =========".format(i))
    for ds_idx, dataset in enumerate(datasets):
      for k_idx, k in enumerate(range(start_k, 12, 2)):
        print("========= dataset={}, k={} =========".format(dataset, k))

        #################################################
        # 成功時，失敗時のトレースをロードし,
        # そこから成功時,失敗時のngramの出現回数を数える
        #################################################

        # pfa spectrumの保存ディレクトリ
        pfa_spec_dir = AbstractData.PFA_SPEC.format(model_type, dataset, i, k, "val")
        # pfaで予測成功時の状態のトレース
        succ_spec_path = os.path.join(pfa_spec_dir, "succ_trace.txt")
        # pfaで予測失敗時の状態のトレース
        fail_spec_path = os.path.join(pfa_spec_dir, "fail_trace.txt")
        # pfaで予測成功/失敗時のトレースを読み込む
        with open(succ_spec_path, 'r') as f:
            succ_rows = f.readlines()
        with open(fail_spec_path, 'r') as f:
            fail_rows = f.readlines()
        # 各行の末尾の改行を取り除き, コンマで配列を切る
        succ_rows = [row.strip().split(',') for row in succ_rows]
        fail_rows = [row.strip().split(',') for row in fail_rows]
        
        for n_idx, ngram_n in enumerate(n_list):
          print(f'============ n={ngram_n} ============')
          s_time = time.perf_counter()
          # 予測に成功/失敗時のn-gramを数える
          succ_ngram_cnt = count_ngram(succ_rows, ngram_n)
          fail_ngram_cnt = count_ngram(fail_rows, ngram_n)

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
            ngram_susp_dir = PfaSusp.NGRAM_SUSP.format(model_type, dataset, i, k, ngram_n, "val")
            save_pickle(os.path.join(ngram_susp_dir, f"{method}_susp_dict.pkl"), ngram_susp_dict[method])
            # 疑惑値の上位20件はすぐみれるようにjsonで保存する
            ngram_susp_dict_top20 = {str(k) : ngram_susp_dict[method][k] for k in list(ngram_susp_dict[method])[:20]}
            save_json(os.path.join(ngram_susp_dir, f"{method}_susp_top20.json"), ngram_susp_dict_top20)
            # print(f'saved in {ngram_susp_dir}')
          f_time = time.perf_counter()
          elapsed_times[i][ds_idx][k_idx][n_idx] = f_time - s_time
          print(f'elapsed:  {elapsed_times[i][ds_idx][k_idx][n_idx]} [sec.]')
  ds_mean = np.mean(elapsed_times, axis=(0, 2, 3))
  with open(f'elapsed_time_{model_type}.csv', 'a') as f:
    for ds, ds_time in zip(datasets, ds_mean):
      f.write(f'{ds}, {ds_time}\n')