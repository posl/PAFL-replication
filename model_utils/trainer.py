import sys
sys.path.append('../') # utilsをインポートするために必要
import copy
import numpy as np
# pytorch関連
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve, roc_auc_score
# 同階層のmodel_utilsからのインポート
from model_utils.gated_rnn import LSTM, GRU, SRNN
from model_utils.model_util import sent2tensor, init_model
# 異なる階層のutilsからのインポート
from utils.constant import *
from utils.help_func import load_pickle, filter_stop_words

def test(data, model, params, mode="test", device="cpu"):
  """
  モデルの評価を行う
  
  Parameters
  ------------------
  data: dict
    データセットのdict
  model: torch.nn.Module
    評価したいモデル
  params: dict
    パラメータのdict
  mode: str
    訓練・テストデータのどちらを使って評価するか
    "train", "test"のどちらかを指定
  device: str
    cpuを使う場合は"cpu"を指定
  
  Returns
  ------------------
  : float
    正解率(=正解数/データ数)
  """
  # 評価用のevalモード(重みを変更しない,ドロップアウトで重みを消さないモード)にする
  model.eval()
  # 訓練・テストデータのどちらで評価するか
  if mode == "train":
    X, Y = data["train_x"], data["train_y"]
  elif mode == "test":
    X, Y = data["test_x"], data["test_y"]
  acc = 0
  for sent, c in zip(X, Y):
    if params["use_clean"]:
      sent = filter_stop_words(sent)
    if params['is_image']:
      input_tensor = torch.unsqueeze(torch.tensor(sent), 0) / 255 # (1, 28, 28)
    else:
      input_tensor = sent2tensor(sent, params["input_size"], data["word_to_idx"], params["WV_MATRIX"], device) # (1, sentの単語数, input_size)
    input_tensor = input_tensor.to(device) # mnistでだけなぜかinput_tensorのdeviceがcpuになってたので追加した
    output, _ = model(input_tensor)
    lasthn = output[0][-1].unsqueeze(0)
    pred = model.h2o(lasthn) # => tensor([[Nのスコア，Pのスコア]])みたいに入ってる
    label = data["classes"].index(c) # data["classes"]=[0,1]の中に含まれるcのインデックス(==c)を返す
    pred = np.argmax(pred.cpu().data.numpy(), axis=1)[0] # predはint型
    acc += 1 if pred == label else 0
  return acc / len(X)

def train(data, params, pre_model=None):
  """
  モデルの訓練を行う.
  テスト精度が最も良かったモデルと,そのモデルの訓練精度,テスト精度を返す.
  
  Parameters
  ------------------
  data: dict
    データセットのdict
    キーは"train_x", "test_x", "train_y", "test_y"など
  params: dict
    学習用のパラメータのdict
  pre_model: torch.nn.Module
    事前に学習済みのモデルを使う場合, この引数に対し, ロードしてきた学習済みモデルを指定する.
    デフォルトはNoneで, この場合, このメソッド内でモデルが新しく作られる.
  
  Returns
  ------------------
  best_model: test_accが最も良かったモデル
  max_train_acc: best_modelのtrain_acc
  max_test_acc: best_modelのtest_acc
  """
  # 事前学習済みのモデルを使わない場合
  if pre_model is None:
    # モデルの初期化
    model = init_model(params)
  # 事前学習済みのモデルを使う場合
  else:
    model = pre_model
  # print(model)
  # モデルをdeviceに渡す
  device = params["device"]
  model = model.to(device)
  # 最適化手法と損失関数の設定
  optimizer = optim.Adadelta(model.parameters(), params["LEARNING_RATE"])
  criterion = nn.CrossEntropyLoss()
  
  pre_test_acc = 0
  max_test_acc = 0
  # エポックのループ
  for e in range(params["EPOCH"]):
    # 訓練データをシャッフル
    data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])
    i = 0
    # trainモードに設定(重みを変更するため)
    model.train()
    # 訓練データの各サンプルに対するループ
    for sent, c in zip(data["train_x"], data["train_y"]):
      # use_cleansが1ならストップワード除去する
      if params["use_clean"]:
        sent = filter_stop_words(sent)
      label = [data["classes"].index(c)]
      label = torch.LongTensor(label).to(device)
      if params['is_image']:
        input_tensor = torch.unsqueeze(torch.tensor(sent), 0) / 255 # (1, 28, 28)
      else:
        # サンプルの文をtensorに変換する(画像データには対応できない)
        input_tensor = sent2tensor(sent, params["input_size"], data["word_to_idx"], params["WV_MATRIX"], device) # (1, sentの単語数, input_size)
      optimizer.zero_grad()
      # モデルの出力を取得
      input_tensor = input_tensor.to(device) # mnistでだけなぜかinput_tensorのdeviceがcpuになってたので追加した
      output, inner_states = model(input_tensor)
      lasthn = output[0][-1].unsqueeze(0)
      pred = model.h2o(lasthn)
      # 損失関数の値を計算
      loss = criterion(pred, label)
      # lossを逆伝搬する(バックプロパゲーション)
      loss.backward()
      # 重みの更新(最適化のステップ)
      optimizer.step()
      # サンプルを500こ処理する度に出力
      if i % 500 == 0:
        print("Train Epoch: {} [{}/{}]\tLoss: {:.6f}".format(e + 1, i + 1, len(data["train_x"]), loss))
      i += 1
    # エポックの終わりにtestする
    test_acc = test(data, model, params, mode="test", device=device)
    print("epoch:", e + 1, "/ test_acc:", test_acc)
    if params["EARLY_STOPPING"] and test_acc <= pre_test_acc:
      print("early stopping by dev_acc!")
      break
    else:
      # 前回のtest_accを更新
      pre_test_acc = test_acc
    # test_accの最大値と,そのときのモデルオブジェクトを更新する
    if test_acc > max_test_acc:
      max_test_acc = test_acc
      best_model = copy.deepcopy(model)
      best_model.i2h.flatten_parameters()
  # best_modelを訓練データに対しても評価する
  max_train_acc = test(data, best_model, params, mode="train", device=device)
  print("train_acc:{0:.4f}, test acc:{1:.4f}".format(max_train_acc, max_test_acc))
  print(best_model)
  return best_model, max_train_acc, max_test_acc

def test4eval(data, model, params, device="cpu", key_x="x", key_y="y"):
  """
  指定したdataに対するmodelの予測を行う.
  testメソッドとの違いは, 様々なメトリクスの測定を目的としている点.
  
  Parameters
  ------------------
  data: dict
    データセットのdict
  model: torch.nn.Module
    評価したいモデル
  params: dict
    パラメータのdict
  device: str
    cpuを使う場合は"cpu"を指定
  key_x: str
    dataにおいて, サンプルが格納されているキー
  key_y: str
    dataにおいて, ラベルが格納されているキー
    
  Returns
  ------------------
  result: dict
    メトリクス名 -> メトリクス値 の対応.
  """

  # 評価用のevalモード(重みを変更しない,ドロップアウトで重みを消さないモード)にする
  model.eval()
  model = model.to(device)
  X, Y = data[key_x], data[key_y]
  # 正解ラベルのリスト
  y_true = Y
  # 予測ラベルのリスト
  y_pred = []
  # ポジティブ確率のリスト
  scores = []

  # (データ,ラベル)1件ずつループ
  for sent, c in zip(X, Y):
    if params['is_image']:
      input_tensor = torch.unsqueeze(torch.tensor(sent), 0) / 255 # (1, 28, 28)
    else:
      input_tensor = sent2tensor(sent, params["input_size"], data["word_to_idx"], params["WV_MATRIX"], device)
    input_tensor = input_tensor.to(device) # mnistでだけなぜかinput_tensorのdeviceがcpuになってたので追加した
    output, _ = model(input_tensor)
    lasthn = output[0][-1].unsqueeze(0)
    # 全結合層の出力(スコア)を取得
    score = model.h2o(lasthn)
    # スコアのlogsoftmaxのexp(=スコアのsoftmax)をとる
    prob = torch.exp(model.softmax(score)) # => tensor([[Nの確率，Pの確率]])に変換する
    # torch.Tensorからnumpy配列に変換
    prob = prob.cpu().data.numpy()
    # モデルの予測ラベルを取り出す(最もスコアの高いラベル)
    pred = np.argmax(prob, axis=1)[0] # predはint型
    # 予測ラベルのリストを更新する
    y_pred.append(pred)
    # auc計算のためのリストを更新する
    if params['is_multiclass']:
      scores.append(prob[0])
    else:
      scores.append(prob[0][1]) # ポジティブの確率をappend
  
  average_mode = 'binary' if not params['is_multiclass'] else 'macro'
  # 各指標の計算
  result = {}
  result['accuracy'] = accuracy_score(y_true, y_pred)
  result['precision'] = precision_score(y_true, y_pred, average=average_mode)
  result['recall'] = recall_score(y_true, y_pred, average=average_mode)
  result['f1_score'] = f1_score(y_true, y_pred, average=average_mode)
  result['mcc'] = matthews_corrcoef(y_true, y_pred)
  # バイナリの分類の場合はauc, roc, 混同行列を記録するが，マルチクラスの場合はaucだけ
  if params['is_multiclass']:
    try:
      result['auc'] = roc_auc_score(y_true, scores, average=average_mode, multi_class='ovo')
    except ValueError as e:
      print(e)
      result['auc'] = -1
  else:
    result['roc'] = roc_curve(y_true, scores)
    # データの全てのラベルが同じだった場合，roc_auc_scoreがValueErrorとなるのでその時は-1にする
    try:
      result['auc'] = roc_auc_score(y_true, scores)
    except ValueError as e:
      print(e)
      result['auc'] = -1
    result['conf_mat'] = confusion_matrix(y_true, y_pred)
  return result

def test_bagging(data, models, params, mode="test", device="cpu"):
  """
  bootsrap sampling aggregationによって，モデルの評価を行う
  
  Parameters
  ------------------
  data: dict
    データセットのdict
  models: list of torch.nn.Module
    評価したいモデルのlist
  params: dict
    パラメータのdict
  mode: str
    訓練・テストデータのどちらを使って評価するか
    "train", "test"のどちらかを指定
  device: str
    cpuを使う場合は"cpu"を指定
  
  Returns
  ------------------
  : float
    正解率(=正解数/データ数)
  """
  # 訓練・テストデータのどちらで評価するか
  if mode == "train":
    X, Y = data["train_x"], data["train_y"]
  elif mode == "test":
    X, Y = data["test_x"], data["test_y"]
  elif mode == "val":
    X, Y = data["val_x"], data["val_y"]

  # baggingに使うモデルの数
  len_bagging = len(models)
  # 正解ラベルのリスト
  y_true = Y
  # 予測ラベルのリスト
  y_pred = []
  # ポジティブ確率のリスト
  pos_score = []

  i = 0
  # データを一件ずつ取り出す
  for sent, c in zip(X, Y):
    # データをtensorに変換する
    input_tensor = sent2tensor(sent, params["input_size"], data["word_to_idx"], params["WV_MATRIX"], device)
    # 集計後の予測確率
    probs = np.ndarray((1,2))
    
    # modelsからモデルを1個ずつ取り出して繰り返す
    for model in models:
      # 評価用のevalモード(重みを変更しない,ドロップアウトで重みを消さないモード)にする
      model.eval()
      # データを入れた時のmodelの出力を得る
      output, _ = model(input_tensor)
      # 最終的な隠れ状態を得る
      lasthn = output[0][-1].unsqueeze(0)
      # 全結合層の出力(スコア)を取得
      score = model.h2o(lasthn)
      # スコアのlogsoftmaxのexp(=スコアのsoftmax)をとる
      prob_tensor = torch.exp(model.softmax(score)) # => tensor([[Nの確率，Pの確率]])に変換する
      # torch.Tensorからnumpy配列に変換
      prob = prob_tensor.cpu().data.numpy()
      probs += prob
      
    # 各モデルの平均の予測確率を全体の予測確率とする
    probs /= len_bagging
    # モデルの予測ラベルを取り出す(0=>N, 1=>P)
    pred = np.argmax(probs, axis=1)[0] # predはint型
    pos_prob = probs[0][1] # ポジティブの確率
    # 予測ラベルのリストとポジティブ確率のリストを更新する
    y_pred.append(pred)
    pos_score.append(pos_prob)
    # サンプルを100こ処理する度に出力
    if i % 100 == 0:
      print("[{}/{}]\t".format(i + 1, len(X)))
    i += 1

  # 各指標の計算
  result = {}
  result['accuracy'] = accuracy_score(y_true, y_pred)
  result['precision'] = precision_score(y_true, y_pred)
  result['recall'] = recall_score(y_true, y_pred)
  result['f1_score'] = f1_score(y_true, y_pred)
  result['mcc'] = matthews_corrcoef(y_true, y_pred)
  result['roc'] = roc_curve(y_true, pos_score)
  # データの全てのラベルが同じだった場合，roc_auc_scoreがValueErrorとなるのでその時はnp.nanにする
  try:
    result['auc'] = roc_auc_score(y_true, pos_score)
  except ValueError as e :
    print(e)
    result['auc'] = np.nan
  result['conf_mat'] = confusion_matrix(y_true, y_pred)
  return result