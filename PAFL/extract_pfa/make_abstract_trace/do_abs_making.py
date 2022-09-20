import glob
import sys
import time
import numpy as np
import argparse
sys.path.append("../../") # extract_pfa/make_abstract_traceをインポートするために必要
sys.path.append("../../../") # model_utils, utilsをインポートするために必要
# 同階層のmake_abstract_traceからのインポート
from PAFL.extract_pfa.make_abstract_trace.make_abs_trace import *
# 異なる階層のutilsからのインポート
from utils.constant import *
from utils.help_func import save_pickle, load_pickle
# 異なる階層のmodel_utilsからのインポート
from model_utils.model_util import load_model, get_model_file

def do_L1_abstract(rnn_traces, rnn_pre, k, data_source, model_type, dataset, partition_type, 
                  save_path, load_model_path, boot_id, output_path=None, partitioner_path=None, num_class=2):
  """
  実際にabstract tracesの生成をおこない, abstract tracesとクラスタリングに用いたpartitionerを保存する.

  Parameters
  ------------------
  rnn_trace: list of str
    original trace
  rnn_pre: list of int
    予測ラベルのリスト
  k: int
    クラスタリングのクラスタ数
  data_source: str
    訓練orテストデータのどちらを入力した時の隠れ状態をクラスタリングするか"train"もしくは"test"を指定.
  model_type: str
    モデルのタイプ. lstm, gruなど. 
  dataset: str
    データセットのタイプ. bp, mrなど.  
  partition_type: str
    クラスタリングに用いるpartitonerのタイプ. 基本Kmenasのみ.
  save_path: str
    abstract tracesの保存パス
  load_model_path: str
    モデルファイルのパス
  boot_id: int
    何番目のbootstrap sampleか (0-indexed)
  """
  # save_pathをルートからのパスに直す
  save_folder = get_path(save_path)
  # abstract tracesの保存パス
  if output_path is None:
    output_path = save_folder.format(model_type, dataset, i, k, data_source + ".txt") if not dataset.isdigit() else \
      save_folder.format(model_type, "tomita_" + dataset, i, k, data_source + ".txt")
  # partitionerの保存パス
  if partitioner_path is None:
    partitioner_path = save_folder.format(model_type, dataset, i, k, data_source + "_partition.pkl") if not dataset.isdigit() else \
      save_folder.format(model_type, "tomita_" + dataset, i, k, data_source + "_partition.pkl")
  # 学習済みのrnnモデルをロード
  rnn = load_model(model_type, dataset, device="cpu", load_model_path=load_model_path)
  s_time =  time.perf_counter()
  # abstract tracesの取得を行う
  abs_seqs, partitioner = level1_abstract(rnn=rnn, rnn_traces=rnn_traces, y_pre=rnn_pre, k=k,
                                          partitioner_exists=False, num_class=num_class,
                                          partition_type=partition_type)
  f_time =  time.perf_counter()
  # abstract tracesを保存
  save_level1_traces(abs_seqs, output_path)
  # クラスタリングに用いたpartitionerはpklで保存
  save_pickle(partitioner_path, partitioner)
  return f_time - s_time # 経過時間を返す

if __name__=="__main__":
  B = 10
  # コマンドライン引数でデータセットやモデルタイプ,GPUの情報を入力
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("--start_boot_id", type=int, help='What id of bootstrap sample starts training from.', default=0)
  args = parser.parse_args()
  dataset, model_type = args.dataset, args.model_type
  isTomita = True if dataset.isdigit() else False
  isImage = True if dataset == DataSet.MNIST else False 
  num_class = 10 if isImage else 2
  # クラスタリングにはKMeansを用いる
  partition_type = PartitionType.KM
  
  elapsed_ik = np.zeros((B, 5))
  for i in range(args.start_boot_id, B):
    print("========= use bootstrap sample {} for training data =========".format(i))
    # モデルファイルが含まれるディレクトリ名を取得
    model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i))  if not isTomita else \
      os.path.join(get_path(getattr(TrainedModel, model_type.upper()).TOMITA).format(dataset), "boot_{}".format(i))
    load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]

    # original traceを読み込むパス
    ori_trace_path = os.path.join(get_path(getattr(getattr(OriTrace, model_type.upper()), dataset.upper())), "boot_{}.pkl".format(i))  if not isTomita else \
      os.path.join(get_path(getattr(OriTrace, model_type.upper()).TOMITA).format(dataset), "boot_{}.pkl".format(i))
    # original traceを読み込む
    print('loading original trace...')
    ori_traces = load_pickle(ori_trace_path)
    # abstract traceを保存するパス(format文字列組み込む前. formatの組み込みはdo_L1_abstract内で行われる)
    save_path = AbstractData.L1

    # クラスタ数は2から10まで2刻みで行う
    for k_idx, k in enumerate(range(2, 12, 2)):
      print("========= k={} =========".format(k))
      # 訓練データを用いたabstract trace生成&保存
      elapsed_ik[i][k_idx] = do_L1_abstract(rnn_traces=ori_traces["train_x"], rnn_pre=ori_traces["train_pre_y"], k=k, 
                      data_source="train", model_type=model_type, dataset=dataset, partition_type=partition_type, 
                      save_path=save_path, load_model_path=load_model_path, boot_id=i, num_class=num_class)
      # テストデータを用いたabstract trace生成&保存
      # do_L1_abstract(rnn_traces=ori_traces["test_x"], rnn_pre=ori_traces["test_pre_y"], k=k, 
      #                 data_source="test", model_type=model_type, dataset=dataset, partition_type=partition_type, 
      #                 save_path=save_path, load_model_path=load_model_path)
      print(f'elapsed = {elapsed_ik[i][k_idx]} [sec.]')
    print("\n")
  print(f'mean elapsed in {B} boot, five k settings: {np.mean(elapsed_ik)} [sec.]')
  with open(f'elapsed_time_{model_type}.csv', 'a') as f:
    f.write(f'{dataset}, {np.mean(elapsed_ik)}\n')