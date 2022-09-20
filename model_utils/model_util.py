import os
import sys
sys.path.append('../') # utilsをインポートするために必要
import torch
from sklearn.metrics import roc_curve, auc
# 同階層のmodel_utilsからのインポート
from model_utils import train_args
from model_utils.gated_rnn import LSTM, GRU, SRNN
# 異なる階層のutilsからのインポート
from utils.constant import *

def init_model(params):
  """
  モデルの初期化を行う
  
  Parameters
  ------------------
  params: dict
    モデルのタイプ, 入力,出力,隠れ状態の次元数などのパラメータ
  
  Returns
  ------------------
  model: torch.nn.Module
    初期化して生成したモデル
  """
  if params["rnn_type"] == ModelType.GRU:
    model = GRU(input_size=params["input_size"], num_class=params["output_size"], 
                hidden_size=params["hidden_size"], num_layers=params["num_layers"])
  elif params["rnn_type"] == ModelType.LSTM:
    model = LSTM(input_size=params["input_size"], num_class=params["output_size"],
                  hidden_size=params["hidden_size"], num_layers=params["num_layers"])
  elif params["rnn_type"] == ModelType.SRNN:
    model = SRNN(input_size=params["input_size"], num_class=params["output_size"],
                  hidden_size=params["hidden_size"], num_layers=params["num_layers"])
  else:
    raise Exception("Unknow rnn type:{}".format(params["rnn_type"]))
  return model

def load_model(model_type, dataset, device, load_model_path):
  """
  モデルをロードして返す
  
  Parameters
  ------------------
  model_type: str
    モデルのタイプ(rnn,lstmなど)
  dataset: str
    データセットのタイプ(bp, mrなど)
  device: str
    cpuを使う場合は"cpu"
    gpuの場合はやってないのでわかりません
  load_model_path: str
    モデルのpklファイルへのパス
  
  Returns
  ------------------
  model: torch.nn.Module
    ロードしたモデルのオブジェクト
  """
  # モデルタイプとデータセットタイプからparamsをロード
  isTomita = True if dataset.isdigit() else False
  params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita else \
    getattr(train_args, "args_{}_tomita".format(model_type))()
  params["rnn_type"] = model_type
  # モデルの初期化を行う
  model = init_model(params=params)

  # 指定したパスに格納されている, モデルの重みをロードしてくる
  # load_state_dictでは重みしか保存しないので最初に初期化が必要
  model_path = get_path(load_model_path)
  model.load_state_dict(torch.load(model_path))
  model = model.to(device)
  model.eval()
  return model

def save_model(save_path, model, train_acc, test_acc):
  """
  学習したモデルを保存する関数
  ファイル名にtrain_accとtest_accをつけてpklで保存する
  
  Parameters
  ------------------
  save_path: str
    モデルを保存するパス
  model: torch.nn.Module
    学習済みモデル
  train_acc: float
    モデルのtrain_acc
  test_acc: float
    モデルのtest_acc
  """
  # save_pathが存在しない場合，そのディレクトリを新たに作る
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  train_acc = "{0:.4f}".format(train_acc)
  test_acc = "{0:.4f}".format(test_acc)
  # train_acc,test_accをファイル名につけて保存
  save_file = os.path.join(save_path, 'train_acc-' + train_acc + '-test_acc-' + test_acc + '.pkl')
  # saveする際はdeviceをcpuに設定しないと, gpuを持たないデバイスで読み込むときにエラーになってしまう
  torch.save(model.cpu().state_dict(), save_file)

def get_model_file(model_type, dataset):
  """
  データセットとモデルの種類から,モデルのpklファイル名を取得する
  
  Parameters
  ------------------
  model_type: str
    モデルのタイプ(rnn,lstmなど)
  dataset: str
    データセットのタイプ(bp, mrなど)
  
  Returns
  ------------------
  model_file: str
    モデルのpklファイル名(ファイル名だけなので,パスではない)
  """
  model_file = ""
  if dataset == DataSet.MR:
    if model_type == ModelType.LSTM:
      model_file = "train_acc-0.7870-test_acc-0.7960.pkl"
    else:
      model_file = ""
  elif dataset == DataSet.BP:
    if model_type == ModelType.LSTM:
      model_file = "train_acc-0.9695-test_acc-0.9530.pkl"
    else:
      model_file = ""
  return model_file

def sent2tensor(sent, input_dim, word2idx, wv_matrix, device):
  """
  文 -> tensorへの変換
  文中の各単語を埋め込み行列によってベクトルに変換する
  
  Parameters
  ------------------
  sent: list of str
    tensorに変換するべき文
  input_dim: int
    RNNへの入力次元数(埋め込みベクトルの次元数)
  word2idx: dict
    単語 -> 単語idへの対応表
  wv_matrix: list of list of float
    単語idから埋め込みベクトルへの対応表(埋め込み行列)
    実態は(語彙数, input_dim)と言う形状の2次元配列
  Returns
  ------------------
  seq: list of list of list of float(torch.tensor)
    各単語を埋め込み行列で変化したベクトルの列
    形状は(1, 文中の単語数, input_dim)という3次元配列
  """
  idx_seq = []
  # 文中に現れる単語のIDの列をidx_seqに格納
  for w in sent:
    if w in word2idx:
      idx = word2idx[w]
    elif w.lower() in word2idx:
      idx = word2idx[w.lower()]
    # w が語彙になかった場合
    else:
      idx = wv_matrix.shape[0] - 1
    idx_seq.append(idx)
  # 3次元の0埋めした tensor を作成(1次元目が必要な理由は不明)
  seq = torch.zeros(1, len(idx_seq), input_dim).to(device)
  # i番目の単語に対応するベクトルを埋め込み行列から取り出す
  for i, w_idx in enumerate(idx_seq):
    seq[0][i] = torch.tensor(wv_matrix[w_idx])
  return seq

def add_data_info(data, params):
  """
  モデルのパラメータを保持するdict変数paramsに3つの情報を付加する
  1)訓練・テストデータに含まれる文の最大の長さ
  2)訓練・テストデータに含まれる文の語彙(単語の種類)数
  3)ラベルの数
  
  Parameters
  ------------------
  data: dict
    データセットのdict
  params: dict
    パラメータのdict
  """
  params["MAX_SENT_LEN"] = max([len(sent) for sent in data["train_x"] + data["test_x"]])
  params["VOCAB_SIZE"] = len(data["vocab"])
  params["CLASS_SIZE"] = len(data["classes"])

def make_y_scores(pos, neg):
  """
  未使用
  get_aucで呼び出される関数
  """
  assert isinstance(pos, list)
  assert isinstance(neg, list)
  scores = pos + neg
  y_ture = [1] * len(pos) + [0] * len(neg)
  return y_ture, scores

def get_auc(pos_score, neg_score):
  """
  未使用
  aucを算出する関数
  """
  y_ture, y_scores = make_y_scores(pos=pos_score, neg=neg_score)
  fpr, tpr, thresholds = roc_curve(y_ture, y_scores)
  auc_score = auc(fpr, tpr)
  return auc_score

