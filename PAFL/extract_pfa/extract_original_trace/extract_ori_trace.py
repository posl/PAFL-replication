import time
import sys
sys.path.append("../../../") # utilsをインポートするために必要
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import sent2tensor, load_model, get_model_file
# 異なる階層のutilsからインポート
from utils.help_func  import load_pickle, save_pickle, filter_stop_words
from utils.constant import *
import torch

def make_ori_trace(model_type, dataset, device, load_model_path, boot_id, use_clean=False, data=None, save_path=None):
  """
  modelをロードし,そのモデルで訓練・テストデータに対するoriginal traceを取る.
  保存先はutils/constant.pyのOriTrceクラスに書いてある.
  
  Parameters
  ------------------
  model_type: str
    モデルのタイプ. lstm, gruなど.
  dataset: str
    データセットのタイプ. mr, bpなど.
  device: str
    CPUを使う場合は"cpu"
  load_model_path: str
    モデルのファイルへのパス
  boot_id: int
    何番目のbootstrap sampleか (0-indexed)
  use_clean: bool
    Trueの場合, filter_stop_wordsする

  Return 
  ------------------
  elapsed: float
  トレース取得にかかった時間[sec.]を返す．
  """
  isTomita = True if dataset.isdigit() else False
  # {boot_id}番目のbootstrap sampleをロード
  print("load data...")
  if data is None:
    data = load_pickle(os.path.join(get_path(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR), "boot_{}.pkl".format(boot_id))) if not isTomita else \
      load_pickle(os.path.join(get_path(DataPath.TOMITA.BOOT_DATA_DIR.format(dataset)), f'boot_{boot_id}.pkl'))
  # 前処理済みのデータをロードしてくる
  ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
    load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)))
  if dataset != DataSet.MNIST:
    word2idx = ori_data["word_to_idx"]
    # word2vecにかけた後のデータをロードしてくる
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita else \
      load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(dataset)))
  
  # 入力する埋め込みベクトルの次元(データセットごとに異なる)
  if dataset == DataSet.BP:
    input_dim = 29
  elif dataset == DataSet.MR or dataset == DataSet.IMDB or dataset == DataSet.TOXIC:
    input_dim = 300
  elif dataset == DataSet.MNIST:
    input_dim = 28
  else: # tomita
    input_dim = 3

  # model_training.pyで学習済みのモデルをロードしてくる
  print("load model...")
  model = load_model(model_type, dataset, device, load_model_path)
  
  # この関数で返すべき変数ori_tracesを初期化
  ori_traces = {}
  ori_traces["train_x"] = []
  ori_traces["test_x"] = []
  ori_traces["train_pre_y"] = []
  ori_traces["test_pre_y"] = []
  
  s_time = time.perf_counter()
  # 訓練データに対するoriginal trace取得
  print("get ori_trace of model for train set...")
  for x in data["train_x"]:
    if dataset == DataSet.MNIST:
      tensor_sequence = torch.unsqueeze(torch.tensor(x), 0) / 255 # (1, 28, 28)
    else:
      if use_clean:
        x = filter_stop_words(x)
      tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
    tensor_sequence = tensor_sequence.to(device)
    hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
    # 途中の隠れ状態のトレースをリストに追加する
    ori_traces["train_x"].append(hn_trace)
    # ラベルは最終的な予測ラベルのみをリストに追加する
    ori_traces["train_pre_y"].append(label_trace[-1])
  
  # テストデータに対するoriginal trace取得
  # print("get ori_trace of model for test set...")
  # for x in ori_data["test_x"]:
  #   if use_clean:
  #     x = filter_stop_words(x)
  #   tensor_sequence = sent2tensor(x, input_dim, word2idx, wv_matrix, device)
  #   hn_trace, label_trace = model.get_predict_trace(tensor_sequence)
  #   # 途中の隠れ状態のトレースをリストに追加する
  #   ori_traces["test_x"].append(hn_trace)
  #   # ラベルは最終的な予測ラベルのみをリストに追加する
  #   ori_traces["test_pre_y"].append(label_trace[-1])

  # ori_tracesを所定のパスに保存して終了
  if save_path is None:
    save_path = os.path.join(get_path(getattr(getattr(OriTrace, model_type.upper()), dataset.upper())), "boot_{}.pkl".format(boot_id)) if not isTomita else \
      os.path.join(get_path(getattr(OriTrace, model_type.upper()).TOMITA.format(dataset)), "boot_{}.pkl".format(boot_id))
  save_pickle(save_path, ori_traces)
  print("original trace saved to {}\n".format(save_path))
  f_time = time.perf_counter()
  elapsed = f_time - s_time
  return elapsed