import os

# プロジェクトルートのパス
PROJECT_ROOT = "/src" # dockerでやるとき
# 学習済みword2vecモデルのパス
WORD2VEC_PATH = "data/wordvec/GoogleNews-vectors-negative300.bin"
# prismの実行ファイル
# PRISM_SCRIPT = "/Users/ishimotoyuuta/OneDrive/prism-4.5-osx64/bin/prism" # localでやるとき
PRISM_SCRIPT = "/home/ubuntu/prism/prism/bin/prism" # awsでやるとき
# prism用の設定ファイル
PROPERTY_FILE = "/prism_api/app/properties.pctl"
START_SYMBOL = 'S'

# 入力パスにプロジェクトルートのパスをくっつけて返す
def get_path(r_path):
  return os.path.join(PROJECT_ROOT, r_path)

# データセット名の定義
class DataSet:
  IMDB = "imdb"
  MR = "mr"
  BP = "bp"
  Tomita1 = "1"
  Tomita2 = "2"
  Tomita3 = "3"
  Tomita4 = "4"
  Tomita5 = "5"
  Tomita6 = "6"
  Tomita7 = "7"

# モデルタイプ名の定義
class ModelType:
  SRNN = 'srnn'
  LSTM = 'lstm'
  GRU = 'gru'

# 各データセットの内容・埋め込み行列の格納パスを定義
class DataPath:
  class BP:
    PROCESSED_DATA = "data/training_data/processed_data/bp/bp.pkl"
    SPLIT_DATA = "data/training_data/split_data/bp/bp.pkl"
    WV_MATRIX = "data/training_data/wv_matrix/bp/wv_matrix_bp.pkl"
    BOOT_DATA_DIR = "data/training_data/bootstrap_data/bp"

  class TOMITA:
    PROCESSED_DATA = "data/training_data/processed_data/tomita/tomita_{}.pkl"
    SPLIT_DATA = "data/training_data/split_data/tomita/tomita_{}.pkl"
    WV_MATRIX = "data/training_data/wv_matrix/tomita/wv_matrix_tomita_{}.pkl"
    BOOT_DATA_DIR = "data/training_data/bootstrap_data/tomita/tomita_{}"

  class IMDB:
    RAW_DATA = "data/training_data/raw_data/imdb"
    PROCESSED_DATA = "data/training_data/processed_data/imdb/processed_imdb.pkl"
    SPLIT_DATA = "data/training_data/split_data/imdb/imdb.pkl"
    WV_MATRIX = "data/training_data/wv_matrix/imdb/wv_matrix_imdb.pkl"
    BOOT_DATA_DIR = "data/training_data/bootstrap_data/imdb"

  class MR:
    RAW_DATA = "data/training_data/raw_data/mr"
    PROCESSED_DATA = "data/training_data/processed_data/mr/processed_mr.pkl"
    SPLIT_DATA = "data/training_data/split_data/mr/mr.pkl"
    WV_MATRIX = "data/training_data/wv_matrix/mr/wv_matrix_mr.pkl"
    BOOT_DATA_DIR = "data/training_data/bootstrap_data/mr"

# 学習済みモデルの格納パスを定義
class TrainedModel:
  class LSTM:
    IMDB = "data/trained_models/lstm/imdb/"
    MR = "data/trained_models/lstm/mr/"
    BP = "data/trained_models/lstm/bp/"  
    TOMITA = "data/trained_models/lstm/tomita_{}/"
  class GRU:
    IMDB = "data/trained_models/gru/imdb/"
    MR = "data/trained_models/gru/mr/"
    BP = "data/trained_models/gru/bp/"
    TOMITA = "data/trained_models/gru/tomita_{}/"
  # 見やすいように整形された予測結果の保存パス(boot_idごとにまとめてある)
  PRED_RESULT_FOR_LOOK = get_path("data/trained_models/pred_table_for_look")

# original trace(RNNの隠れ状態のトレース)の格納パスを定義
class OriTrace:
  class LSTM:
    IMDB = "data/original_trace/lstm/imdb/"
    MR = "data/original_trace/lstm/mr/"
    BP = "data/original_trace/lstm/bp/"
    TOMITA = "data/original_trace/lstm/tomita_{}/"
  class GRU:
    IMDB = "data/original_trace/gru/imdb"
    MR = "data/original_trace/gru/mr"
    BP = "data/original_trace/gru/bp"
    TOMITA = "data/original_trace/gru/tomita_{}/"

# クラスタリングのタイプを定義
class PartitionType:
  KM = "km"  # kmeans
  KMP = "kmp"  # kmeans based on probas
  HC = "hc"  # hierarchical-clustering

# abstract trace, pfa関連の保存ディレクトリを定義
class AbstractData:
  # abstract tracesの保存ディレクトリ
  # data/abstract_trace/{model_type}/{dataset}/boot_{boot_id}/k={k}/{data_source}.txt
  L1 = get_path("data/abstract_trace/{}/{}/boot_{}/k={}/{}")
  # pfaの保存ディレクトリ
  L2 = get_path("data/pfa_model/{}/{}/boot_{}/k={}/alpha={}")
  # pfa spectrumの保存ディレクトリ
  PFA_SPEC = get_path("data/pfa_spectrum/{}/{}/boot_{}/k={}/{}")
  # pfa上でのトレース(txt)の保存ディレクトリ
  # フォーマットはdata/pfa_trace/{model_type}/{dataset}/boot_{boot_id}/k={k}/{train|test}
  PFA_TRACE = get_path("data/pfa_trace/{}/{}/boot_{}/k={}")

# PFAの疑惑値を格納するディレクトリを定義
class PfaSusp:
  TRANS_SUSP = get_path("data/pfa_susp/pfa_trans_susp/{}/{}/boot_{}/k={}/{}")
  STATE_SUSP = get_path("data/pfa_susp/pfa_state_susp/{}/{}/boot_{}/k={}/{}")
  NGRAM_SUSP = get_path("data/pfa_susp/pfa_ngram_susp/{}/{}/boot_{}/k={}/n={}/{}")

# 抽出したデータを格納するディレクトリを定義
class ExtractData:
  DIR = get_path("data/extracted_data/{}/{}/boot_{}/k={}/n={}/theta={}")
  # 抽出したデータ自体の保存パス
  EX = get_path("data/extracted_data/{}/{}/boot_{}/k={}/n={}/theta={}/ex_data.pkl")
  RAND = get_path("data/extracted_data/{}/{}/boot_{}/k={}/n={}/theta={}/rand_data.pkl")
  # 抽出したデータを予測した結果のdictの保存パス
  EX_PRED = get_path("data/extracted_data/{}/{}/boot_{}/k={}/n={}/theta={}/ex_pred_result.pkl")
  RAND_PRED = get_path("data/extracted_data/{}/{}/boot_{}/k={}/n={}/theta={}/rand_pred_result.pkl")
  # 見やすいように整形された予測結果の保存パス(theta, nの値ごとにまとめてある)
  PRED_RESULT_FOR_LOOK = get_path("data/extracted_data/pred_table_for_look/boot_{}/n={}_theta={}.csv")

class RetrainedModel:
  # 抽出, ランダムデータそれぞれで再学習したモデル及びテストデータに対する評価メトリクスの保存ディレクトリ
  EX = get_path("data/retrained_model/{}/{}/k={}/theta={}/ex")
  RAND = get_path("data/retrained_model/{}/{}/k={}/theta={}/rand")
  VAL = get_path("data/retrained_model/{}/{}/val")
  # 見やすいように整形された予測結果の保存パス(thetaの値ごとにまとめてある)
  PRED_RESULT_FOR_LOOK = get_path("data/retrained_model/pred_table_for_look/theta={}")