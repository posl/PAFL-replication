import sys
sys.path.append('../') # utilsをインポートするため必要
import gensim
# 同階層data_preparationからのインポート
from data_preparation.prepare_mr import divide_mr
from data_preparation.prepare_imdb import divide_imdb, full_imdb
from data_preparation.prepare_bp import divide_bp
from data_preparation.prepare_tomita import divide_tomita
from data_preparation.split_data import data_split4extract
# 異なる階層utilsからのインポート
from utils.constant import WORD2VEC_PATH, get_path, DataSet, DataPath
from utils.help_func import save_pickle, load_pickle

if __name__ == "__main__":
  # データセットの種類はコマンドライン引数で受け取る
  dataset = sys.argv[1]
  isTomita = False

  if dataset == DataSet.MR:
    # word2vecの読み込み
    word2vec_model_path = get_path(WORD2VEC_PATH)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=True)
    # データの前処理を行い，その結果と単語ベクトルをpklファイルとして保存する．
    processed_data = divide_mr(word2vec_model)
    # processed_data = load_pickle(get_path(DataPath.MR.PROCESSED_DATA))
  
  elif dataset == DataSet.IMDB:
    # word2vecの読み込み
    word2vec_model_path = get_path(WORD2VEC_PATH)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=True)
    # データの前処理を行い，その結果と単語ベクトルをpklファイルとして保存する．
    full_imdb(word2vec_model)
    processed_data = divide_imdb()
    # processed_data = load_pickle(get_path(DataPath.IMDB.PROCESSED_DATA))

  elif dataset == DataSet.BP:
    processed_data = divide_bp()
    # processed_data = load_pickle(get_path(DataPath.BP.PROCESSED_DATA))
  
  else:
    isTomita = True
    processed_data = divide_tomita(dataset)
    # processed_data = load_pickle(get_path(DataPath.TOMITA.PROCESSED_DATA.format(dataset)))
  
  # processed_dataを, train, val, testに分割したsplit_dataにする
  split_data = data_split4extract(processed_data, data_source="train", ratio=0.25)
  # 分割後のデータを保存
  save_pickle(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA), split_data) if not isTomita else \
    save_pickle(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset)), split_data)
  print("saved in {}".format(get_path(getattr(DataPath, dataset.upper()).SPLIT_DATA))) if not isTomita else \
    print("saved in {}".format(get_path(DataPath.TOMITA.SPLIT_DATA.format(dataset))))
  