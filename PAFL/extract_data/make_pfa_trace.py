import os
import sys
import argparse
import glob
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
# 異なる階層のget_pfa_spectrum, 同じ階層のextract_dataからインポートするために必要
sys.path.append("../")
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model
# 異なる階層のget_pfa_spectrumからインポート
from get_pfa_spectrum.do_pfa_spectrum import get_total_symbols, load_model_and_data
from get_pfa_spectrum.trace_on_pfa import trace_abs_seq, extract_l1_trace

def abstract_trace_by_train_partition(model_type, dataset, boot_id, k, model, data, input_dim, file_name, key, num_class, use_clean, wv_matrix=None):
  """
  対象となるデータを対象となるRNNに入力した際の隠れ状態を, trainデータで学習したpartitionerによってクラスタに割り当てる
  1) データやRNNモデル, partitionerのロード
  2) keyで指定されるデータデータの各サンプルについて以下を行う
      隠れ状態のトレースを取得
      得られたトレースに含まれる各隠れ状態をpartitionerによってクラスタに割り当てる
      ファイルに追記
  Parameters
  --------------------------
  model_type: str
    モデルのタイプ. lstm, gru.
  dataset: str
    データセットのタイプ. mr, bp.
  boot_id: int
    何番目のbootstrap samplingを使うか．
  k: int
    PFA構築の際のクラスタ数.
  model: torch.nn.Module
    学習済みRNNモデル
  data: dict
    対象となるデータセット.
    このdataをmodelに入力した際の隠れ状態のトレースをクラスタに割り当てる.
  wv_matrix: list of list of float
    単語の埋め込み行列.
  input_dim: int
    modelの入力次元数.
  file_name: str
    保存するabstract tracesのtxtファイルのファイル名
  key: str
    辞書型変数dataの中で，, 抽出対象となるキーを表す文字列
    data['test_x], data['test_y]を対象にしたいなら'test'を指定.
  """
  # partitionerのパスを指定し, ロード
  pt_path = AbstractData.L1.format(model_type, dataset, boot_id, k, "train_partition.pkl")
  partitioner = load_pickle(pt_path)
  # abs_traceの保存パス
  save_path = AbstractData.L1.format(model_type, dataset, boot_id, k, file_name + ".txt")
  # 対象データの各サンプルについて, 指定したpartitionerによってabs traceを取得
  X, Y = data["{}_x".format(key)], data["{}_y".format(key)]
  for x, y in zip(X, Y):
    if dataset == DataSet.MNIST:
      abs_trace = extract_l1_trace(x, model, input_dim, "cpu", partitioner, use_clean=use_clean, num_class=num_class)
    else:
      abs_trace = extract_l1_trace(x, model, input_dim, "cpu", partitioner, word2idx=data["word_to_idx"], wv_matrix=wv_matrix, use_clean=use_clean, num_class=num_class)
    with open(save_path, "a") as f:
      line = ",".join([str(ele) for ele in abs_trace])
      f.write(line + "\n")

def load_abs_data(model_type, dataset, boot_id, k, abs_trace_name, total_symbols):
  """
  モデルやデータセット, クラスタ数の情報から, クラスタリングのpartitioner, pfaの遷移関数を取り出して返す.
  
  Parameters
  ------------------
  model_type: str
    モデルのタイプ. lstm, gru.
  dataset: str
    データセットのタイプ. mr, bp.
  boot_id: int
    何番目のbootstrap samplingを使うか．
  k: int
    PFA構築の際のクラスタ数.
  abs_trace_name: str
    abstract tracesのファイル名(拡張子を除いた部分)
  total_symbols: int
    PFA構築の際のtotal_symbols.pfaのファイル名についてる数字.

  Returns
  ------------------
  abstract_traces: list of list of str
    各サンプルのabstract traceたち
  trans_func: list of list of str
    現在の状態, 入力シンボル から, 次の状態への対応を格納する行列
  trans_wfunc: list of list of float
    遷移確率の行列. 
  """
  # abstract tracesのパス
  abs_trace_path = AbstractData.L1.format(model_type, dataset, boot_id, k, abs_trace_name + ".txt")
  # abstract tracesをロード
  with open(abs_trace_path, "r") as f:
    abs_traces = f.readlines()
  # pfaの遷移関数のファイルのパス
  trans_func_dir = AbstractData.L2.format(model_type, dataset, boot_id, k, 64)
  trans_func_path = os.path.join(trans_func_dir, "train_{}_transfunc.pkl").format(total_symbols)
  # pfaの遷移関数をロード
  dfa = load_pickle(get_path(trans_func_path))
  trans_func, trans_wfunc = dfa["trans_func"], dfa["trans_wfunc"]
  return abs_traces, trans_func, trans_wfunc  

if __name__ == "__main__":
  B = 10
  device = "cpu"
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("i", type=int, help='The id of bootstrap sample.')
  parser.add_argument("k", type=int, help='The number of cluster.')
  args = parser.parse_args()
  dataset, model_type, i, k = args.dataset, args.model_type, args.i, args.k
  
  isTomita = dataset.isdigit()
  # 変数datasetを変更する(tomitaの場合，"1" => "tomita_1"にする)
  if isTomita:
    tomita_id = dataset
    dataset = "tomita_" + tomita_id
  isImage = (dataset == DataSet.MNIST)
  input_type = 'text' if not isImage else 'image'
  num_class = 10 if isImage else 2
  
  # オリジナルのデータの読み込み
  ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
    load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(tomita_id)))
  if not isImage:
    # wv_matrixとinput_dimの設定
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita else \
      load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(tomita_id)))
  else:
    wv_matrix = None

  if isTomita:
    input_dim = 3
  elif isImage:
    input_dim = 28
  elif dataset == DataSet.BP:
    input_dim = 29
  else: # dataset == MR or IMDB
    input_dim = 300

  if dataset == DataSet.MR or dataset == DataSet.IMDB or dataset == DataSet.TOXIC:
    use_clean = 1
  else:
    use_clean = 0

  # valデータの情報
  target_source = "val"
  abs_trace_file_name = "{}_by_train_partition".format(target_source)

  # for i in range(1, B):
  print("========= boot_id={}, k={} =========".format(i, k))
  # pfaファイルのロードに必要な情報
  total_symbols = get_total_symbols(dataset, model_type, i) if not isTomita else \
    get_total_symbols(tomita_id, model_type, i)

  # データを読み出す
  data = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "{}_boot_{}.pkl".format(target_source, i)))) if not isTomita else \
  load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(tomita_id), "{}_boot_{}.pkl".format(target_source, i))))
  # train_x, train_y, val_x, val_y以外の属性は合わせる
  keys = list(ori_data.keys()).copy()
  keys.remove('train_x'); keys.remove('train_y')
  keys.remove('val_x'); keys.remove('val_y')
  for key in keys:
    data[key] = ori_data[key]

  # モデルを読み出す
  model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
    os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(tomita_id), "boot_{}".format(i))
  load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
  model = load_model(model_type, dataset, device, load_model_path) if not isTomita else \
    load_model(model_type, tomita_id, device, load_model_path)

  # 抽出対象となるvalデータのabstract tracesをとっておく
  # すでに存在する場合は削除してしまう
  if os.path.exists(AbstractData.L1.format(model_type, dataset, i, k, abs_trace_file_name + '.txt')):
    os.remove(AbstractData.L1.format(model_type, dataset, i, k, abs_trace_file_name + '.txt'))
  print("make abstract traces for val data...")
  abstract_trace_by_train_partition(model_type, dataset, i, k, model, data, input_dim, abs_trace_file_name, target_source, num_class, use_clean, wv_matrix)
  print("done making abstract traces for val data.")
  # for abs_trace_name in ["train", "test_by_train_partition", abs_trace_file_name]:
  for abs_trace_name in [abs_trace_file_name]:
    # pfa tracesの保存ディレクトリ
    pfa_trace_dir = AbstractData.PFA_TRACE.format(model_type, dataset, i, k)
    os.makedirs(pfa_trace_dir, exist_ok=True)
    # pfa tracesの保存パス
    pfa_trace_path = os.path.join(pfa_trace_dir, abs_trace_name + ".txt")
    # 既にpfa traceのテキストファイルが存在する場合は一回削除する
    os.remove(pfa_trace_path) if os.path.exists(pfa_trace_path) else None
    # 必要なabstract tracesや遷移関数をロード
    abs_traces, trans_func, trans_wfunc = load_abs_data(model_type, dataset, i, k, abs_trace_name, total_symbols)
    
    # abs_tracesの各トレースを, pfaトレースに変換していく
    for abs_trace in abs_traces:
      abs_trace = abs_trace.split(",")
      # 初期状態を"S"でなく"1"に変える
      abs_trace[0] = "1"
      # 末尾の改行文字を取り除く
      abs_trace[-1] = abs_trace[-1].replace('\n', '')
      # abs_traceとtrans_funcからpfa_traceを構成, 第二引数の遷移の行列は取らない
      pfa_trace, _ = trace_abs_seq(abs_trace, trans_func, trans_wfunc, total_states=10**4)
      # pfa traceを追記していく
      with open(pfa_trace_path, "a") as f:
        f.write((','.join([str(l) for l in pfa_trace]))+'\n')
    
    print("PFA traces Saved in {}.".format(pfa_trace_path))