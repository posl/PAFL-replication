import sys
sys.path.append("../../") # extract_pfa/learning_pfaからインポートするために必要
sys.path.append("../../../") # utilsをインポートするために必要
# 同じ階層learning_pfaからのインポート
from PAFL.extract_pfa.learning_pfa.aalergia import *
from PAFL.extract_pfa.learning_pfa.read_abs_data import load_trace_data
# 異なる階層のutilsからのインポート
from utils.constant import *
from utils.time_util import current_timestamp

def do_L2_abstract(k, data_source, model_type, dataset, boot_id, total_symbols=1000000, alpha=64, abs_trace_path=None):
  """
  引数から特定されるabstract tracesを読み込んで, そこからpfaを構成する. 
  出力されるのは以下の2つで, 同じディレクトリ内に作成される.
  1) pfaを表すpmファイル
  2) pfaの遷移関数を表すpklファイル
  
  Parameters
  ------------------
  k: int
    abstract traces構成に用いたクラスタ数.
  data_source: str
    "train" or "test"
  model_type: str
    モデルのタイプ. lstm, gruなど.
  dataset: str
    データセットのタイプ. mr, bpなど.
  total_symbols: int
    合計何文字のシンボルをabstract tracesから読み込むか. デフォルト=1000000. 
  alpha: int
    AALERGIAアルゴリズムのパラメータ. デフォルト=64. 
  """

  # abstract traceが保存されているディレクトリのパス
  if abs_trace_path is None:
    abs_trace_path = AbstractData.L1.format(model_type, dataset, boot_id, k, data_source + ".txt")
  # pfaのpmファイルと遷移関数のpklファイルを保存するディレクトリのパス
  output_path = AbstractData.L2.format(model_type, dataset, boot_id, k, alpha)

  print("=======k={}=======".format(k))
  # abstract traceを読み込む
  sequence, alphabet = load_trace_data(abs_trace_path, total_symbols)
  # AALERGIAクラスのインスタンス作成
  print("{}, init".format(current_timestamp()))
  al = AALERGIA(alpha, sequence, alphabet, start_symbol=START_SYMBOL, output_path=output_path,
                show_merge_info=False)
  # AALERGIAのアルゴリズムを適用してpfaを構成する
  print("{}, learing....".format(current_timestamp()))
  dffa = al.learn()
  print("{}, done.".format(current_timestamp()))
  # 構成したpfaのpmファイルと遷移関数のpklファイルを出力して終わり
  al.output_prism(dffa, data_source)


if __name__=="__main__":
  B = 10
  # コマンドライン引数からデータセットとモデルのタイプ, データソース("train" or "test")を受け取る
  dataset = sys.argv[1]
  dataset = "tomita_{}".format(dataset) if dataset.isdigit() else dataset
  model_type = sys.argv[2]
  # using_train_set = int(sys.argv[3]) # train setを使う場合1, test setの場合0
  using_train_set = True
  data_source = "train" if using_train_set else "test"

  for i in range(B):
    print("========= use bootstrap sample {} for training data =========".format(i))
    for k in range(2, 12, 2):
      do_L2_abstract(k, data_source, model_type, dataset, i)
    print("\n")