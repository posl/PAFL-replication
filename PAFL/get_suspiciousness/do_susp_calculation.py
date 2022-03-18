import os
import sys
#同じ階層のディレクトリからインポートするために必要
sys.path.append("../")
# 異なる階層のutilsをインポートするために必要
sys.path.append("../../")
# 同じ階層のget_suspeciousnessからインポート
from PAFL.get_suspiciousness.calc_trans_susp import *
from PAFL.get_suspiciousness.calc_state_susp import get_state_susp
# 異なる階層utilsからインポート
from utils.constant import *
from utils.help_func import load_pickle, save_pickle, save_json, load_json

def rank_trans_susp(trans_susp_mat):
  """
  遷移ごとの疑惑値の行列から, (遷移前状態, 遷移後状態) => その遷移の疑惑値の対応辞書を作る.
  辞書は疑惑値の降順でソートする.
  
  Parameters
  ------------------
  trans_susp_mat: list of list of float
    対象となる行列（Ochiaiの行列など）
  
  Returns
  ------------------
  trans_susp_dict: dict
    (遷移前状態, 遷移後状態) => その遷移の疑惑値 の対応辞書. 疑惑値の降順でソートされている.
  """
  # 引数trans_susp_matは(状態数)*(状態数)のnumpy2次元配列になっているはず. それの確認.
  assert trans_susp_mat.shape[0] == trans_susp_mat.shape[1]
  total_states = trans_susp_mat.shape[0]
  
  trans_susp_dict = dict()
  for i in range(total_states):
    for j in range(total_states):
      # (遷移前状態, 遷移後状態) => その遷移の疑惑値 の対応辞書を作る
      trans_susp_dict[(i+1, j+1)] = trans_susp_mat[i][j]
  # 遷移の疑惑値の降順で辞書をソートして返す
  # sortedの返り値はlistになるので注意
  trans_susp_dict = dict(sorted(trans_susp_dict.items(), key=lambda x:x[1], reverse=True))
  return trans_susp_dict

if __name__ == "__main__":
  B = 10

  # コマンドライン引数から受け取り
  dataset = sys.argv[1]
  model_type = sys.argv[2]
  # どのデータから疑惑値を計算するか
  target_source = "val"

  for i in range(B):
    print("========= use bootstrap sample {} for training data =========".format(i))

    for k in range(2, 12, 2):
      # 遷移ごと/状態ごとの疑惑値を保存するディレクトリ
      trans_susp_save_dir = PfaSusp.TRANS_SUSP.format(model_type, dataset, i, k, target_source)
      state_susp_save_dir = PfaSusp.STATE_SUSP.format(model_type, dataset, i, k, target_source)
      # pfa spectrumの格納ディレクトリ
      pfa_spec_dir = AbstractData.PFA_SPEC.format(model_type, dataset, i, k, target_source)
      # pred_infoの保存パス
      pred_info_path = os.path.join(pfa_spec_dir, "pred_info.json")
      # pred_info.jsonをdict型で読み込む
      pred_info = load_json(pred_info_path)

      # 予測に成功/失敗したサンプル数を取得
      Ns, Nf = count_succ_fail(pred_info)
      # pfa spectrumをロード
      succ_mat, fail_mat = load_pickle(os.path.join(pfa_spec_dir, "succ_pass_mat.pkl")), \
                          load_pickle(os.path.join(pfa_spec_dir, "fail_pass_mat.pkl"))
      
      # 3つのsbfl手法をそれぞれ適応して疑惑値を計算し保存する
      for method in [calc_ochiai, calc_tarantula, calc_dstar]:
        # ochiai, tarantula, dstarなど手法名だけ取得
        method_name = method.__name__.split('_')[1]
        
        # 遷移ごとの疑惑値の行列を計算
        trans_susp_mat = get_trans_susp_mat(Ns, Nf, succ_mat, fail_mat, method)
        # 行列形式からソートされた辞書形式に変換
        trans_susp_dict = rank_trans_susp(trans_susp_mat)
        # json保存用に疑惑値上位20件だけ取り出す
        trans_susp_dict4json = {str(k) : trans_susp_dict[k] for k in list(trans_susp_dict)[:20]}
        # trans_susp_matをpklで保存する
        save_pickle(os.path.join(trans_susp_save_dir, method_name+"_susp_mat.pkl"), trans_susp_mat)
        # trans_susp_dictをpklで保存する
        save_pickle(os.path.join(trans_susp_save_dir, method_name+"_susp_dict.pkl"), trans_susp_dict)
        # trans_susp_dict4jsonをjsonで保存する
        save_json(os.path.join(trans_susp_save_dir, method_name+"_susp_top20.json"), trans_susp_dict4json)
        
        # 状態ごとの疑惑値を計算
        state_susp_dict = get_state_susp(trans_susp_mat)
        # json保存用に疑惑値上位20件だけ取り出す
        state_susp_dict4json = {str(k) : state_susp_dict[k] for k in list(state_susp_dict)[:20]}
        # state_susp_dictをpklで保存する
        save_pickle(os.path.join(state_susp_save_dir, method_name+"_susp_dict.pkl"), state_susp_dict)
        # state_susp_dict4jsonをjsonで保存する
        save_json(os.path.join(state_susp_save_dir, method_name+"_susp_top20.json"), state_susp_dict4json)