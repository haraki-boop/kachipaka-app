import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd
import lightgbm as lgb

INPUT_CSV = "ml_target_data.csv"
MODEL_FILE = "keiba_ai_model.pkl"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip()

def get_dist_cat(d):
    if pd.isna(d): return np.nan
    if d <= 1400: return 'sprint'
    elif d <= 2200: return 'mile_middle'
    else: return 'stayer'

def preprocess_features(df):
    df_feat = df.copy()

    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    df_feat = df_feat.dropna(subset=['date_parsed']).sort_values(['馬名_clean', 'date_parsed'])

    df_feat['distance_num'] = pd.to_numeric(df_feat.get('distance'), errors='coerce')
    df_feat['dist_cat'] = df_feat['distance_num'].apply(get_dist_cat)
    df_feat['rank_num'] = pd.to_numeric(df_feat.get('着順'), errors='coerce')
    df_feat['is_top3_past'] = (df_feat['rank_num'] <= 3).astype(int)
    df_feat['is_win_past'] = (df_feat['rank_num'] == 1).astype(int)

    # 距離適性特徴量の算出（過去のデータのみ参照）
    dist_diffs = []
    cat_win_rates = []
    cat_runs_list = []

    # 馬ごとに時系列処理
    for horse, group in df_feat.groupby('馬名_clean', sort=False):
        group_indices = group.index
        for i in range(len(group)):
            curr_row = group.iloc[i]
            past_rows = group.iloc[:i] # 前走までのデータ

            curr_dist = curr_row['distance_num']
            curr_cat = curr_row['dist_cat']

            if past_rows.empty:
                dist_diffs.append(np.nan)
                cat_win_rates.append(np.nan)
                cat_runs_list.append(0)
            else:
                # 好走距離の平均との差
                top3_past = past_rows[past_rows['is_top3_past'] == 1]
                if not top3_past.empty and pd.notna(curr_dist):
                    best_dist_avg = top3_past['distance_num'].mean()
                    dist_diffs.append(abs(curr_dist - best_dist_avg))
                else:
                    dist_diffs.append(np.nan)

                # 同カテゴリ内の成績
                cat_past = past_rows[past_rows['dist_cat'] == curr_cat]
                c_runs = len(cat_past)
                cat_runs_list.append(c_runs)
                if c_runs > 0:
                    cat_win_rates.append(cat_past['is_win_past'].sum() / c_runs)
                else:
                    cat_win_rates.append(np.nan)

    df_feat['dist_diff'] = dist_diffs
    df_feat['cat_win_rate'] = cat_win_rates
    df_feat['cat_runs'] = cat_runs_list

    # レース間隔
    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days

    # 近走パフォーマンス（過去3走平均）
    target_cols = ['my_time_idx', 'my_last3f_idx', 'my_pace_idx', 'my_start_idx']
    for col in target_cols:
        if col in df_feat.columns:
            num_col = pd.to_numeric(df_feat[col], errors='coerce')
            df_feat[f'eff_{col}'] = num_col.groupby(df_feat['馬名_clean']).apply(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
        else:
            df_feat[f'eff_{col}'] = np.nan

    # 前走情報
    df_feat['prize_num'] = pd.to_numeric(df_feat.get('賞金(万円)'), errors='coerce')
    df_feat['prev_prize'] = df_feat.groupby('馬名_clean')['prize_num'].shift(1)
    df_feat['prev_rank_num'] = df_feat['rank_num'].groupby(df_feat['馬名_clean']).shift(1)

    # 騎手・馬実績
    j_col = df_feat.get('jockey_win_power', df_feat.get('jockey_win_rate', pd.Series()))
    df_feat['eff_jockey_win'] = pd.to_numeric(j_col, errors='coerce').clip(0.0, 1.0)
    df_feat['eff_jockey_track_win'] = pd.to_numeric(df_feat.get('jockey_track_win_rate'), errors='coerce').clip(0.0, 1.0)
    df_feat['horse_win_rate_val'] = pd.to_numeric(df_feat.get('horse_win_rate'), errors='coerce').clip(0.0, 1.0)
    df_feat['horse_runs_val'] = pd.to_numeric(df_feat.get('horse_runs'), errors='coerce')

    return df_feat

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print("Loading historical data...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='cp932')

    if '着順' not in df.columns:
        print("Error: '着順' column missing.")
        return

    df['is_win'] = (pd.to_numeric(df['着順'], errors='coerce') == 1).astype(int)
    df_clean = df.dropna(subset=['is_win']).copy()
    
    print("Processing features with Distance Aptitude...")
    df_prep = preprocess_features(df_clean)

    candidate_features = [
        '馬番', '枠番', '斤量', 'distance_num',
        'dist_diff', 'cat_win_rate', 'cat_runs', # 新規追加
        'interval_days', 'prev_prize', 'prev_rank_num', 
        'horse_win_rate_val', 'horse_runs_val',
        'eff_my_time_idx', 'eff_my_last3f_idx', 'eff_my_pace_idx', 'eff_my_start_idx',
        'eff_jockey_win', 'eff_jockey_track_win'
    ]

    use_features = [f for f in candidate_features if f in df_prep.columns]
    
    cat_cols = ['surface_code', 'condition_code', 'sex_code']
    for cat in cat_cols:
        if cat in df_prep.columns:
            df_prep[cat] = df_prep[cat].astype('category')
            use_features.append(cat)

    X = df_prep[use_features].copy()
    y = df_prep['is_win']

    print(f"Training LightGBM model for WIN probability with {len(X)} records...")
    
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(params, train_data, num_boost_round=150)
    joblib.dump({'model': model, 'features': use_features}, MODEL_FILE)
    print(f"Success! Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()