import os
import sys
import argparse
import glob
import torch
import time
# 異なる階層のutils, model_utilsをインポートするために必要
sys.path.append("../../")
import numpy as np
# 異なる階層のutilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle
# 異なる階層のmodel_utilsからインポート
from model_utils.model_util import load_model, get_model_file, sent2tensor
from model_utils import train_args
from model_utils.trainer import test4eval
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_curve, roc_auc_score

if __name__ == "__main__":
  B = 10
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("--start_boot_id", type=int, help='What boo_id starts from.', default=0)
  parser.add_argument("--start_k", type=int, help='What k starts from.', default=2)
  parser.add_argument("--start_n", type=int, help='What n starts from.', default=1)
  args = parser.parse_args()
  dataset, model_type = args.dataset, args.model_type
  start_boot_id, start_k, start_n = args.start_boot_id, args.start_k, args.start_n

  isTomita = dataset.isdigit()
  # 変数datasetを変更する(tomitaの場合，"1" => "tomita_1"にする)
  if isTomita:
    tomita_id = dataset
    dataset = "tomita_" + tomita_id
  isImage = (dataset == DataSet.MNIST)
  num_class = 10 if isImage else 2

  print("\ndataset = {}\nmodel_type = {}".format(dataset, model_type))
  
  # 埋め込み行列のロード
  if not isImage:
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita  else \
    load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(tomita_id)))

  # オリジナルのデータの読み込み
  ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
    load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(tomita_id)))

  for i in range(start_boot_id, B):
    print("\n============== boot_id={} ==============".format(i))
    # bootがきまったら検証データを最初に全部予測して，予測結果のリストを作っておく
    # そのあと，抽出インデックスをロードしてきて一部選択すれば，モデルで予測するのは各bootで1回で済む

    # モデルを読み出す
    model_dir = os.path.join(getattr(getattr(TrainedModel, model_type.upper()), dataset.upper()), "boot_{}".format(i)) if not isTomita  else \
      os.path.join(getattr(TrainedModel, model_type.upper()).TOMITA.format(tomita_id), "boot_{}".format(i))
    load_model_path = glob.glob(get_path(os.path.join(model_dir, "*.pkl")))[0]
    model = load_model(model_type, dataset, device, load_model_path) if not isTomita else \
      load_model(model_type, tomita_id, device, load_model_path)
    # bootstrap samplingをロード
    boot = load_pickle(get_path(os.path.join(getattr(DataPath, dataset.upper()).BOOT_DATA_DIR, "{}_boot_{}.pkl".format('val', i)))) if not isTomita else \
      load_pickle(get_path(os.path.join(DataPath.TOMITA.BOOT_DATA_DIR.format(tomita_id), "{}_boot_{}.pkl".format('val', i))))
    val_x, val_y = boot['val_x'], boot['val_y']
    # モデルのパラメータを保持しておく
    params = getattr(train_args, "args_{}_{}".format(model_type, dataset))() if not isTomita else \
      getattr(train_args, "args_{}_{}".format(model_type, "tomita"))()
    params["device"] = device
    params["is_image"] = isImage
    params["is_multiclass"] = True if num_class > 2 else False
    if not isImage:
      params["WV_MATRIX"] = wv_matrix

    # 予測結果のリストを返すコードをかいちゃうよ〜
    val_pred = [] # モデルが予測したラベル等
    scores = []   # auc計算のためのスコア
    model.eval()
    for sent, c in zip(val_x, val_y):
      if params['is_image']:
        input_tensor = torch.unsqueeze(torch.tensor(sent), 0) / 255 # (1, 28, 28)
      else:
        input_tensor = sent2tensor(sent, params["input_size"], ori_data["word_to_idx"], params["WV_MATRIX"], device) # (1, sentの単語数, input_size)
      input_tensor = input_tensor.to(device) # mnistでだけなぜかinput_tensorのdeviceがcpuになってたので追加した
      output, _ = model(input_tensor)
      lasthn = output[0][-1].unsqueeze(0)
      score = model.h2o(lasthn) # => tensor([[Nのスコア，Pのスコア]])みたいに入ってる
      prob = torch.exp(model.softmax(score)) # => tensor([[Nの確率，Pの確率]])に変換する
      # torch.Tensorからnumpy配列に変換
      prob = prob.cpu().data.numpy()
      # auc計算のため
      if num_class == 2:
        scores.append(prob[0][1]) # 2値分類の場合はポジティブの確率
      else:
        scores.append(prob[0])    # 多値分類の場合は確率の配列
      pred = np.argmax(prob, axis=1)[0] # predはint型
      val_pred.append(pred)

    for n in range(start_n, 6):
      for k in range(start_k, 12, 2):
        for theta in [10, 50]:
          # 抽出 / ランダムデータのインデックスのパス
          indices_path = os.path.join(ExtractData.DIR.format(model_type, dataset, i, k, n, theta), 'names_indices.npz')
          names_indices = np.load(indices_path, allow_pickle=True)
          names, indices_list, rand_indices = names_indices['names'], names_indices['indices'], names_indices['rand']
          for m_idx, method_name in enumerate([*names, 'rand']):
            print(f'--------------------------------------------\nboot {i}, n={n}, k={k}, theta={theta}, method={method_name}')
            # インデックスをロード
            if method_name == 'rand':
              indices = rand_indices
            else:
              indices = indices_list[m_idx]
            # 真のラベルと予測ラベルのうちインデックスで指定した分だけ抽出
            y_true, y_pred = np.array(val_y)[indices], np.array(val_pred)[indices]
            # auc計算用のスコアも抽出
            val_scores = np.array(scores)[indices]
            # 各種メトリクスを算出  
            average_mode = 'binary' if not params['is_multiclass'] else 'macro'
            result = {}
            result['accuracy'] = accuracy_score(y_true, y_pred)
            result['precision'] = precision_score(y_true, y_pred, average=average_mode)
            result['recall'] = recall_score(y_true, y_pred, average=average_mode)
            result['f1_score'] = f1_score(y_true, y_pred, average=average_mode)
            result['mcc'] = matthews_corrcoef(y_true, y_pred)
            # バイナリの分類の場合はauc, roc, 混同行列を記録するが，マルチクラスの場合はaucだけ
            if params['is_multiclass']:
              try:
                result['auc'] = roc_auc_score(y_true, val_scores, average=average_mode, multi_class='ovo')
              except ValueError as e:
                print(e)
                result['auc'] = np.nan
            else:
              result['roc'] = roc_curve(y_true, val_scores)
              # データの全てのラベルが同じだった場合，roc_auc_scoreがValueErrorとなるのでその時は-1にする
              try:
                result['auc'] = roc_auc_score(y_true, val_scores)
              except ValueError as e:
                print(e)
                result['auc'] = np.nan
              result['conf_mat'] = confusion_matrix(y_true, y_pred)
            # 結果を一応表示しておく
            if not params["is_multiclass"]:
              print(result["conf_mat"])
            else:
              print(f'acc.: {result["accuracy"]}, mcc.: {result["mcc"]}')
            # 抽出データと同じディレクトリに, result(dict型)をjsonとして保存する
            save_pickle(os.path.join(ExtractData.DIR.format(model_type, dataset, i, k, n, theta), "{}_pred_result.pkl".format(method_name)), result)
    print(f'boot {i} is finished!!!') 