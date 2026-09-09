import pandas as pd
import numpy as np
import os
import joblib
import re
import unicodedata

INPUT_CSV = "ml_target_data.csv"
OUTPUT_CSV = "ml_target_data_v2.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
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

    print("前処理および特徴量生成を実行中...")
    df_feat = df.copy()

    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    df_feat = df_feat.dropna(subset=['date_parsed', '着順'])
    df_feat['着順'] = pd.to_numeric(df_feat['着順'], errors='coerce')
    df_feat = df_feat.dropna(subset=['着順'])

    df_feat['distance_num'] = pd.to_numeric(df_feat.get('distance'), errors='coerce')
    df_feat['dist_cat'] = df_feat['distance_num'].apply(get_dist_cat)
    df_feat['place_code_str'] = df_feat.get('place_code', df_feat['race_id'].astype(str).str[4:6]).astype(str)
    df_feat['rank_num'] = df_feat['着順']
    df_feat['is_win'] = (df_feat['rank_num'] == 1).astype(int)

    time_col = df_feat.get('タイム', df_feat.get('time', df_feat.get('time_seconds', pd.Series(np.nan, index=df_feat.index))))
    df_feat['time_sec_clean'] = time_col.apply(parse_time_str)
    
    last3f_col = df_feat.get('上り', df_feat.get('上がり', df_feat.get('上がり3F', df_feat.get('last_3f_val', pd.Series(np.nan, index=df_feat.index)))))
    df_feat['last3f_sec_clean'] = last3f_col.apply(parse_time_str)

    passing_results = df_feat['通過'].apply(parse_passing).tolist()
    df_feat['first_pos_clean'] = [p[0] for p in passing_results]
    df_feat['last_pos_clean'] = [p[1] for p in passing_results]
    df_feat['corner_diff'] = [p[2] for p in passing_results]

    drop_cols = ['race_avg_time', 'race_std_time', 'race_avg_last3f', 'race_std_last3f', 'race_avg_pos', 'race_std_pos']
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    race_stats = df_feat.groupby('race_id').agg(
        race_avg_time=('time_sec_clean', 'mean'),
        race_std_time=('time_sec_clean', 'std'),
        race_avg_last3f=('last3f_sec_clean', 'mean'),
        race_std_last3f=('last3f_sec_clean', 'std'),
        race_avg_pos=('first_pos_clean', 'mean'),
        race_std_pos=('first_pos_clean', 'std')
    ).reset_index()
    df_feat = pd.merge(df_feat, race_stats, on='race_id', how='left')

    race_std_time = df_feat['race_std_time'].replace(0, np.nan)
    race_std_last3f = df_feat['race_std_last3f'].replace(0, np.nan)
    race_std_pos = df_feat['race_std_pos'].replace(0, np.nan)

    df_feat['my_time_idx'] = 50.0 + ((df_feat['race_avg_time'] - df_feat['time_sec_clean']) / race_std_time) * 10.0
    df_feat['my_last3f_idx'] = 50.0 + ((df_feat['race_avg_last3f'] - df_feat['last3f_sec_clean']) / race_std_last3f) * 10.0
    df_feat['my_start_idx'] = 50.0 + ((df_feat['race_avg_pos'] - df_feat['first_pos_clean']) / race_std_pos) * 10.0

    df_feat['my_time_idx'] = df_feat['my_time_idx'].fillna(50.0)
    df_feat['my_last3f_idx'] = df_feat['my_last3f_idx'].fillna(50.0)
    df_feat['my_start_idx'] = df_feat['my_start_idx'].fillna(50.0)

    # ステップ1: 【純粋日付ソート】
    df_feat = df_feat.sort_values(['date_parsed', 'race_id']).reset_index(drop=True)

    if '騎手' in df_feat.columns:
        df_feat['jockey_clean'] = df_feat['騎手'].astype(str).str.strip()
        df_feat['jockey_win_rate'] = df_feat.groupby('jockey_clean')['is_win'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.1)

    if '調教師' in df_feat.columns:
        df_feat['trainer_win_rate'] = df_feat.groupby('調教師')['is_win'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0.08)

    # ステップ2: 【馬名・時系列ソート】
    df_feat = df_feat.sort_values(['馬名_clean', 'date_parsed']).reset_index(drop=True)

    df_feat['prev_dist'] = df_feat.groupby('馬名_clean')['distance_num'].shift(1)
    df_feat['dist_change_num'] = df_feat['distance_num'] - df_feat['prev_dist'].fillna(df_feat['distance_num'])
    
    df_feat['same_dist_avg_rank'] = df_feat.groupby(['馬名_clean', 'dist_cat'])['rank_num'].transform(lambda x: ewm_shift(x, 3)).fillna(7.0)
    
    df_feat['surface_avg_rank'] = df_feat.groupby(['馬名_clean', 'surface'])['rank_num'].transform(lambda x: ewm_shift(x, 3)).fillna(7.0)
    df_feat['is_heavy_track'] = df_feat['condition'].astype(str).apply(lambda x: 'heavy' if x in ['重', '不良', '稍重'] else 'good')
    df_feat['condition_avg_rank'] = df_feat.groupby(['馬名_clean', 'is_heavy_track'])['rank_num'].transform(lambda x: ewm_shift(x, 3)).fillna(7.0)
    df_feat['turn_direction'] = df_feat['place_code_str'].apply(lambda x: 'left' if x in ['04', '05', '07'] else 'right')
    df_feat['turn_avg_rank'] = df_feat.groupby(['馬名_clean', 'turn_direction'])['rank_num'].transform(lambda x: ewm_shift(x, 3)).fillna(7.0)

    df_feat['eff_my_start_idx'] = df_feat.groupby('馬名_clean')['my_start_idx'].transform(lambda x: ewm_shift(x, 3)).fillna(50.0)
    df_feat['eff_my_last3f_idx'] = df_feat.groupby('馬名_clean')['my_last3f_idx'].transform(lambda x: ewm_shift(x, 3)).fillna(50.0)
    
    # 🚀【増殖ロジック1】長期実力と短期調子の差分（モメンタム）
    df_feat['eff_my_start_idx_long'] = df_feat.groupby('馬名_clean')['my_start_idx'].transform(lambda x: ewm_shift(x, 10)).fillna(50.0)
    df_feat['eff_my_last3f_idx_long'] = df_feat.groupby('馬名_clean')['my_last3f_idx'].transform(lambda x: ewm_shift(x, 10)).fillna(50.0)
    df_feat['start_idx_trend'] = df_feat['eff_my_start_idx'] - df_feat['eff_my_start_idx_long']
    df_feat['last3f_idx_trend'] = df_feat['eff_my_last3f_idx'] - df_feat['eff_my_last3f_idx_long']

    prize_col = '賞金(万円)' if '賞金(万円)' in df_feat.columns else 'prize'
    df_feat['prize_num'] = pd.to_numeric(df_feat.get(prize_col, 0), errors='coerce').fillna(0.0)
    df_feat['horse_prize_avg'] = df_feat.groupby('馬名_clean')['prize_num'].transform(lambda x: ewm_shift(x, 5)).fillna(0.0)
    
    df_feat['race_avg_prize'] = df_feat.groupby('race_id')['horse_prize_avg'].transform('mean').replace(0, 1)
    df_feat['race_prize_relative'] = df_feat['horse_prize_avg'] / df_feat['race_avg_prize']
    df_feat['race_prize_rank'] = df_feat.groupby('race_id')['horse_prize_avg'].rank(ascending=False, method='min')

    df_feat['prev_prize'] = df_feat.groupby('馬名_clean')['prize_num'].shift(1).fillna(0.0)
    df_feat['prize_diff_vs_prev'] = df_feat['race_avg_prize'] - df_feat['prev_prize']

    df_feat['prev_1c'] = df_feat.groupby('馬名_clean')['first_pos_clean'].shift(1).fillna(10.0)
    df_feat['prev_last_corner'] = df_feat.groupby('馬名_clean')['last_pos_clean'].shift(1).fillna(10.0)
    df_feat['prev_corner_diff'] = df_feat.groupby('馬名_clean')['corner_diff'].shift(1).fillna(0.0)
    
    df_feat['prev_rank'] = df_feat.groupby('馬名_clean')['rank_num'].shift(1).fillna(7.0)
    df_feat['prev2_rank'] = df_feat.groupby('馬名_clean')['rank_num'].shift(2).fillna(7.0)
    df_feat['momentum_rank'] = df_feat['prev2_rank'] - df_feat['prev_rank']

    if 'jockey_clean' in df_feat.columns:
        df_feat['prev_jockey'] = df_feat.groupby('馬名_clean')['jockey_clean'].shift(1)
        df_feat['is_jockey_change'] = (df_feat['jockey_clean'] != df_feat['prev_jockey']).astype(int)

    df_feat['kinryo_num'] = pd.to_numeric(df_feat.get('斤量'), errors='coerce').fillna(55.0)
    weights_parsed = df_feat.get('馬体重', pd.Series()).apply(parse_weight)
    df_feat['body_weight'] = [p[0] for p in weights_parsed]
    df_feat['kinryo_weight_ratio'] = df_feat['kinryo_num'] / df_feat['body_weight'].fillna(470)
    
    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days.fillna(30)
    df_feat['is_fresh'] = (df_feat['interval_days'] >= 60).astype(int)
    df_feat['is_tight'] = (df_feat['interval_days'] <= 21).astype(int)

    # ステップ3: 【レース単位での集計処理】
    df_feat['race_expected_pace'] = df_feat.groupby('race_id')['prev_1c'].transform('mean')

    # 🚀【増殖ロジック2】同じレースを走る他馬との相対化（偏差値・差分化）
    # AIに「この馬は今日のメンバーの中で相対的にどうなのか」を直感的に理解させる
    base_cols_for_relative = [
        'kinryo_num', 'body_weight', 'interval_days', 
        'eff_my_start_idx', 'eff_my_last3f_idx', 
        'jockey_win_rate', 'trainer_win_rate', 'horse_prize_avg'
    ]
    for col in base_cols_for_relative:
        if col in df_feat.columns:
            mean_s = df_feat.groupby('race_id')[col].transform('mean')
            std_s = df_feat.groupby('race_id')[col].transform('std').replace(0, 1)
            df_feat[f'{col}_race_diff'] = df_feat[col] - mean_s
            df_feat[f'{col}_race_zscore'] = (df_feat[col] - mean_s) / std_s

    df_feat = df_feat.sort_values(['date_parsed', 'race_id', '着順']).reset_index(drop=True)

    drop_temp_cols = ['is_heavy_track', 'turn_direction', 'prev_jockey']
    df_feat = df_feat.drop(columns=[c for c in drop_temp_cols if c in df_feat.columns])

    print(f"データを保存中: {OUTPUT_CSV}...")
    df_feat.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("【修正完了】最強の戦術特徴量および相対化データを大量追加しました！")

if __name__ == "__main__":
    main()