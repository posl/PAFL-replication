def load_trace_data(abs_trace_path, symbols_count, start_symbol=None):
  """
  abstract traceをよみこみ, list型の変数として再構成して返す.
  また, abstract traceに含まれるシンボルの集合も返す.
  
  Parameters
  ------------------
  abs_trace_path: str
    abstract tracesの保存先のパス
  symbols_count: int
    合計何文字のシンボルをabstract tracesから読み込むか.
    symbols_count以上は読み込まない.
  start_symbol: str, default:None
    特定の開始シンボルがあれば指定する. デフォルトはNone.
  
  Returns
  ------------------
  seq_list: list of lost of str
  alphabet: set of str
  """
  seq_list = []
  alphabet = set()
  # cnt: 読み取ったシンボルの合計数
  cnt = 0
  with open(abs_trace_path, 'rt') as f:
    for line in f.readlines():
      line = line.strip().strip("'").strip(",")
      # 各行をコンマで分割し配列に入れる
      seq = line.split(",")
      # remain_len: 読み取れるシンボル数の残り
      remain_count = symbols_count - cnt
      # 現在の行すべて読める場合
      if remain_count >= len(seq):
        cnt += len(seq)
      # 現在の行をすべて読み込むと読み取った数がsymbols_countを超えてしまう場合
      else:
        seq = seq[:remain_count]
        cnt += remain_count
      seq = [start_symbol] + seq if start_symbol is not None else seq
      # 各行のトレースを配列に追加していく
      seq_list.append(seq)
      # シンボルの集合の更新
      alphabet = alphabet.union(set(seq))
      if symbols_count != -1 and cnt >= symbols_count:
        break
    if cnt < symbols_count:
        print("no enough data, actual load {} symbols".format(cnt))
    return seq_list, alphabet
