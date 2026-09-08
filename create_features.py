import pandas as pd
import numpy as np
import os
import joblib
import re

# 🌟 実際の環境に合わせてファイル名を指定
INPUT_CSV = "ml_target_data.csv"
OUTPUT_CSV = "ml_target_data_v2.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    import unicodedata
    s = unicodedata.normalize('NFKC', str(name))
    s = re.sub(r'[\s・･.\-ー_]+', '', s).strip()
    return s.upper()

def get_dist_cat(d):
    if pd.isna(d): return np.nan
    if d <= 1400: return 'sprint'
    elif d <= 2200: return 'mile_middle'
    else: return 'stayer'

def parse_weight(val):
    if pd.isna(val): return np.nan, np.nan
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m: return float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0
    return np.nan, np.nan

def parse_passing(val):
    if pd.isna(val): return np.nan, np.nan, np.nan
    parts = str(val).split('-')
    try: return float(parts[0]), float(parts[-1]), float(parts[0]) - float(parts[-1])
    except: return np.nan, np.nan, np.nan

def parse_time_str(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    m = re.match(r'(?:(\d+)[:.])?(\d{1,2})\.(\d+)', s)
    if m:
        mins = int(m.group(1)) if m.group(1) else 0
        secs = int(m.group(2))
        ms = float('0.' + m.group(3))
        return mins * 60 + secs + ms
    try: return float(s)
    except: return np.nan

# 🌟 魔改造: EMA（指数平滑移動平均）＋ リーク防止の shift(1)
def ewm_shift(x, span):
    return x.shift(1).ewm(span=span, min_periods=1).mean()

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} が見つかりません。")
        return

    print("データを読み込み中...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='cp932')

    print("魔改造EMA ＆ データリーク完全防止 前処理を実行中...")
    df_feat = df.copy()

    # --- 1. 基本的なクレンジングとソート（時系列順） ---
    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    df_feat = df_feat.dropna(subset=['date_parsed', '着順'])
    df_feat['着順'] = pd.to_numeric(df_feat['着順'], errors='coerce')
    df_feat = df_feat.dropna(subset=['着順'])
    
    # 🌟 リーク防止の鉄則: ここで必ず「馬名」と「日付」でソートする
    df_feat = df_feat.sort_values(['馬名_clean', 'date_parsed']).reset_index(drop=True)

    df_feat['distance_num'] = pd.to_numeric(df_feat.get('distance'), errors='coerce')
    df_feat['dist_cat'] = df_feat['distance_num'].apply(get_dist_cat)
    df_feat['place_code_str'] = df_feat.get('place_code', df_feat['race_id'].astype(str).str[4:6]).astype(str)
    df_feat['rank_num'] = df_feat['着順']
    df_feat['is_win'] = (df_feat['rank_num'] == 1).astype(int)

    # --- 2. タイムと指数のベース計算 ---
    time_col = df_feat.get('タイム', df_feat.get('time', pd.Series(np.nan, index=df_feat.index)))
    df_feat['time_sec_clean'] = time_col.apply(parse_time_str)
    
    last3f_col = df_feat.get('上がり3F', df_feat.get('上がり', df_feat.get('last3f', pd.Series(np.nan, index=df_feat.index))))
    df_feat['last3f_sec_clean'] = last3f_col.apply(parse_time_str)

    passing = df_feat['通過'].apply(parse_passing)
    df_feat['first_pos_clean'] = [p[0] for p in passing]
    df_feat['last_pos_clean'] = [p[1] for p in passing]
    df_feat['corner_diff'] = [p[2] for p in passing]

    # 🌟 エラー解消: すでに計算済みの列が残っていると、merge時に _x, _y がついてエラーになるため削除
    drop_cols = ['race_avg_time', 'race_std_time', 'race_avg_last3f', 'race_std_last3f', 'race_avg_pos', 'race_std_pos']
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # --- 3. レースごとの平均値を計算（今回のレースのレベルを測るため） ---
    race_stats = df_feat.groupby('race_id').agg(
        race_avg_time=('time_sec_clean', 'mean'),
        race_std_time=('time_sec_clean', 'std'),
        race_avg_last3f=('last3f_sec_clean', 'mean'),
        race_std_last3f=('last3f_sec_clean', 'std'),
        race_avg_pos=('first_pos_clean', 'mean'),
        race_std_pos=('first_pos_clean', 'std')
    ).reset_index()
    df_feat = pd.merge(df_feat, race_stats, on='race_id', how='left')

    # 標準偏差が0またはNaNの場合のゼロ除算対策
    race_std_time = df_feat['race_std_time'].replace(0, np.nan)
    race_std_last3f = df_feat['race_std_last3f'].replace(0, np.nan)
    race_std_pos = df_feat['race_std_pos'].replace(0, np.nan)

    df_feat['my_time_idx'] = 50.0 + ((df_feat['race_avg_time'] - df_feat['time_sec_clean']) / race_std_time) * 10.0
    df_feat['my_last3f_idx'] = 50.0 + ((df_feat['race_avg_last3f'] - df_feat['last3f_sec_clean']) / race_std_last3f) * 10.0
    df_feat['my_start_idx'] = 50.0 + ((df_feat['race_avg_pos'] - df_feat['first_pos_clean']) / race_std_pos) * 10.0

    # NaNを50.0で埋める
    df_feat['my_time_idx'] = df_feat['my_time_idx'].fillna(50.0)
    df_feat['my_last3f_idx'] = df_feat['my_last3f_idx'].fillna(50.0)
    df_feat['my_start_idx'] = df_feat['my_start_idx'].fillna(50.0)

    # --- 4. 【魔改造＆リーク防止】 過去の成績から「前走までの実績」を生成 ---
    # 必ず .shift(1) を使い、今回の結果が混ざらないようにする
    df_feat['prev_dist'] = df_feat.groupby('馬名_clean')['distance_num'].shift(1)
    df_feat['dist_change_num'] = df_feat['distance_num'] - df_feat['prev_dist'].fillna(df_feat['distance_num'])
    
    # 🌟 魔改造: 着順の安定感を「直近3走（EMA）」で評価
    df_feat['same_dist_avg_rank'] = df_feat.groupby(['馬名_clean', 'dist_cat'])['rank_num'].transform(lambda x: ewm_shift(x, 3)).fillna(7.0)
    df_feat['same_place_avg_rank'] = df_feat.groupby(['馬名_clean', 'place_code_str'])['rank_num'].transform(lambda x: ewm_shift(x, 3)).fillna(7.0)

    # 🌟 魔改造: テンの速さ・末脚のキレも「直近3走（EMA）」で評価
    df_feat['eff_my_start_idx'] = df_feat.groupby('馬名_clean')['my_start_idx'].transform(lambda x: ewm_shift(x, 3)).fillna(50.0)
    df_feat['eff_my_last3f_idx'] = df_feat.groupby('馬名_clean')['my_last3f_idx'].transform(lambda x: ewm_shift(x, 3)).fillna(50.0)
    
    # 🌟 魔改造: 賞金も「直近5走（EMA）」の勢いで相対評価
    prize_col = '賞金(万円)' if '賞金(万円)' in df_feat.columns else 'prize'
    df_feat['prize_num'] = pd.to_numeric(df_feat.get(prize_col, 0), errors='coerce').fillna(0.0)
    df_feat['horse_prize_avg'] = df_feat.groupby('馬名_clean')['prize_num'].transform(lambda x: ewm_shift(x, 5)).fillna(0.0)
    
    df_feat['race_avg_prize'] = df_feat.groupby('race_id')['horse_prize_avg'].transform('mean').replace(0, 1)
    df_feat['race_prize_relative'] = df_feat['horse_prize_avg'] / df_feat['race_avg_prize']
    
    # 通過順位の過去情報
    df_feat['prev_1c'] = df_feat.groupby('馬名_clean')['first_pos_clean'].shift(1).fillna(10.0)
    df_feat['prev_last_corner'] = df_feat.groupby('馬名_clean')['last_pos_clean'].shift(1).fillna(10.0)
    df_feat['prev_corner_diff'] = df_feat.groupby('馬名_clean')['corner_diff'].shift(1).fillna(0.0)

    # --- 5. その他基礎指標 ---
    df_feat['kinryo_num'] = pd.to_numeric(df_feat.get('斤量'), errors='coerce').fillna(55.0)
    weights_parsed = df_feat.get('馬体重', pd.Series()).apply(parse_weight)
    df_feat['body_weight'] = [p[0] for p in weights_parsed]
    df_feat['kinryo_weight_ratio'] = df_feat['kinryo_num'] / df_feat['body_weight'].fillna(470)
    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days.fillna(30)
    
    # --- 6. カテゴリ変数のエンコーディング ---
    print("カテゴリ情報を数値に変換中...")
    if 'surface' in df_feat.columns:
        from sklearn.preprocessing import LabelEncoder
        le_surf = LabelEncoder()
        df_feat['surface_code'] = le_surf.fit_transform(df_feat['surface'].astype(str))
        joblib.dump(le_surf, "le_surf.pkl")
        
    if 'condition' in df_feat.columns:
        from sklearn.preprocessing import LabelEncoder
        le_cond = LabelEncoder()
        df_feat['condition_code'] = le_cond.fit_transform(df_feat['condition'].astype(str))
        joblib.dump(le_cond, "le_cond.pkl")

    # 元のレース順に戻す
    df_feat = df_feat.sort_index()

    print(f"データを保存中: {OUTPUT_CSV}...")
    df_feat.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("【大成功】データリークを完全防止し、近走重視（EMA）を組み込んだAI用データが完成しました！")
    print("※ 必ず学習スクリプト（train_xxx.py）を再度実行して、AIモデルを最新化してください！")

if __name__ == "__main__":
    main()