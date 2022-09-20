import os
import sys 
import time
import numpy as np
import argparse
#同じ階層のディレクトリからインポートするために必要
sys.path.append("../")
# 異なる階層のmodel_utils, utilsをインポートするために必要
sys.path.append("../../")
import shutil
import glob
# 異なる階層utilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_json
# 異なる階層model_utilsからインポート
from model_utils.model_util import load_model, get_model_file
# 同じ階層get_pfa_spectrumからインポート
from PAFL.get_pfa_spectrum.get_reachability_prob import prepare_prism_data
from PAFL.get_pfa_spectrum.trace_on_pfa import test_acc_fdlt

def load_model_and_data(dataset, model_type):
  """
  データセットとモデルの種類から, 必要なモデルやデータなどを読み込んで返す.
  データは分割済みのsplit dataをロードする.  
  Parameters
  ------------------
  dataset: str
    データセットの種類. bp, mrなど. 
  model_type: str
    モデルの種類. lstm, gruなど.
  
  Returns
  ------------------
  model: torch.nn.Module
    学習済みのRNNモデル.
  data: dict
    訓練・テストデータセットなどからなるdict. 
  wv_matrix: list of list of float
    単語の埋め込み行列.
  input_dim: int
    入力ベクトルの次元数
  """
  # データセットとモデルの種類からモデルファイル名を取得
  model_file = get_model_file(model_type, dataset)
  # モデルファイルが含まれるディレクトリ名を取得
  model_dir = getattr(getattr(TrainedModel, model_type.upper()), dataset.upper())
  # モデルファイルへのパスを取得
  load_model_path = get_path(os.path.join(model_dir, model_file))
  # モデルのロード
  model = load_model(model_type, dataset, device="cpu", load_model_path=load_model_path)
  # データセットのロード
  data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA))
  # 埋め込み行列のロード
  wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX))
  # データセットに応じたinput_dimの設定
  if dataset == DataSet.BP:
    input_dim = 29
  else:
    input_dim = 300
  return model, data, wv_matrix, input_dim

def get_total_symbols(dataset, model_type, boot_id, k=2, alpha=64, data_source="train"):
  """
  pmファイルのファイル名をパースして合計シンボル数を返す.
  
  Parameters
  ------------------
  dataset: str
    データセットの種類.
  model_type: str
    モデルの種類.
  boot_id: int
    使用したbootstrap samplingのid 
  k: int
    abstractionの際のクラスタ数.
  alpha: int
    AALERGIAのパラメータ.
  data_source: str
    訓練・テストのどちらのデータセットを用いて作ったpfaを参照するか. "train" もしくは "test"を指定.
  
  Returns
  ------------------
  total_symbols: int
    pfa構築の際の合計シンボル数
  """
  isTomita = dataset.isdigit()
  # pfaのpmファイルなどの保存ディレクトリのパス
  dfa_dir = get_path(AbstractData.L2.format(model_type, dataset, boot_id, k, alpha)) if not isTomita else \
    get_path(AbstractData.L2.format(model_type, "tomita_" + dataset, boot_id, k, alpha))
  # dfa_dir内のpmファイルのパスを取得
  file_fullname = glob.glob(os.path.join(dfa_dir, '*.pm'))[0]
  # フルパスからファイル名の部分だけ取得
  file_name = os.path.basename(file_fullname)
  # ファイル名の部分から拡張子以外の部分だけ取得
  file_name = os.path.splitext(file_name)[0]
  # 拡張子以外の部分からtotal_symbolの部分だけ取得して返す
  total_symbols = int(file_name.split('_')[1])
  return total_symbols

if __name__=="__main__":
  B = 10
  device = "cpu"
  # コマンドライン引数から受け取り
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", type=str, help='abbrev. of datasets')
  parser.add_argument("model_type", type=str, help='type of models')
  parser.add_argument("--start_boot_id", type=int, help='What id of bootstrap sample starts training from.', default=0)
  parser.add_argument("--start_k", type=int, help='What k starts training from.', default=2)
  args = parser.parse_args()
  dataset, model_type = args.dataset, args.model_type

  print(f'dataset = {dataset}, model_type = {model_type}')

  isTomita = dataset.isdigit()
  isImage = (dataset == DataSet.MNIST)
  input_type = 'text' if not isImage else 'image'
  num_class = 10 if isImage else 2
  # 変数datasetを変更する(tomitaの場合，"1" => "tomita_1"にする)
  if isTomita:
    tomita_id = dataset
    dataset = "tomita_" + tomita_id
  
  # オリジナルのデータの読み込み
  ori_data = load_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA)) if not isTomita else \
    load_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(tomita_id)))
  if not isImage:
    # wv_matrixとinput_dimの設定
    wv_matrix = load_pickle(get_path(getattr(DataPath, dataset.upper()).WV_MATRIX)) if not isTomita else \
      load_pickle(get_path(DataPath.TOMITA.WV_MATRIX.format(tomita_id)))

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

  # pfa構築に用いたデータソース(train|test)
  data_source = "train"
  # pfa構築に用いたalphaの値
  alpha = 64
  # pfa spectrumを取得するデータソース
  target_source = "val"
  
  # 経過時間を保持する配列
  elapsed_ik = np.zeros((B, 5))

  # 予測ラベルを読みだす
  val_pred_path = f'../train_rnn/val_pred_labels_{model_type}.pkl'
  val_pred_labels_dic = load_pickle(val_pred_path)

  # bootidでループ
  for i in range(args.start_boot_id, B):
    init_k = args.start_k if i == args.start_boot_id else 2
    print("========= use bootstrap sample {} for training data =========".format(i))
    total_symbols = get_total_symbols(dataset, model_type, i)
    
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
    model.eval()

    if isTomita:
      val_pred_labels = val_pred_labels_dic[f'{tomita_id}_{i}']
    else:
      val_pred_labels = val_pred_labels_dic[f'{dataset}_{i}']

    for k_idx, k in enumerate(range(init_k, 12, 2)):
      print("========= k={} =========".format(k))
      # PFAの予測時のログなどを格納するディレクトリのパスを設定
      save_path = AbstractData.PFA_SPEC.format(model_type, dataset, i, k, target_source)
      os.makedirs(save_path, exist_ok=True)

      # partitionerのパスを指定し, ロードする
      pt_path = AbstractData.L1.format(model_type, dataset, i, k, data_source + "_partition.pkl")
      partitioner = load_pickle(pt_path)
      # pfaの格納ディレクトリのパスを指定
      dfa_file_dir = AbstractData.L2.format(model_type, dataset, i, k, alpha)
      # pfaの遷移関数のpklファイルのパスを指定し, ロードする
      trans_func_path = os.path.join(dfa_file_dir, "{}_{}_transfunc.pkl").format(data_source, total_symbols)
      dfa = load_pickle(trans_func_path)
      # pfaのpmファイルのパスを指定
      pm_file_path = os.path.join(dfa_file_dir, "{}_{}.pm").format(data_source, total_symbols)
      
      # reachability計算用のpmファイル群を生成
      total_states, tmp_prism_data = prepare_prism_data(pm_file_path, num_prop=num_class)
      s_time = time.perf_counter()
      # PFAによる予測の実行
      if input_type == 'text':
        pred_info = test_acc_fdlt(X=data["{}_x".format(target_source)], Y=data["{}_y".format(target_source)],
                                  model=model, partitioner=partitioner, dfa=dfa,
                                  tmp_prism_data=tmp_prism_data, input_type=input_type,
                                  word2idx=data["word_to_idx"], wv_matrix=wv_matrix, use_clean=use_clean,
                                  input_dim=input_dim,device="cpu", total_states=total_states, save_path=save_path, num_class=num_class, val_pred_labels=val_pred_labels)
      else:
        pred_info = test_acc_fdlt(X=data["{}_x".format(target_source)], Y=data["{}_y".format(target_source)],
                                  model=model, partitioner=partitioner, dfa=dfa,
                                  tmp_prism_data=tmp_prism_data, input_type=input_type, use_clean=use_clean,
                                  input_dim=input_dim,device="cpu", total_states=total_states, save_path=save_path, num_class=num_class, val_pred_labels=val_pred_labels)
      save_json(os.path.join(save_path, "pred_info.json"), pred_info)
      f_time = time.perf_counter()
      elapsed_ik[i][k_idx] = f_time - s_time
      print(
          "k={}\t#states={}\trnn_acc={:.4f}\tpfa_acc={:.4f}\tfdlt={:.4f}\tunspecified:{}/{}".format(
            k, pred_info["total_states"], pred_info["rnn_acc"], pred_info["pfa_acc"], pred_info["fdlt"], pred_info["unspecified"], len(data["{}_y".format(target_source)])))
      # PFAでの予測が終わったらtmp_prism_dataのディレクトリは不要なので削除
      shutil.rmtree(tmp_prism_data)
      print(f'saved in {os.path.join(save_path, "pred_info.json")}')
      print(f'elapsed = {elapsed_ik[i][k_idx]} [sec.]')
  elapsed_mean = np.mean(elapsed_ik)
  print(f'mean elapsed in {B} boot, five k settings: {elapsed_mean} [sec.]')
  with open(f'elapsed_time_{model_type}.csv', 'a') as f:
    f.write(f'{dataset}, {elapsed_mean}\n')