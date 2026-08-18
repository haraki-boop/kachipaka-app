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

    # コースIDの作成（例: 札幌_芝_1800m）
    if 'place_code' in df_feat.columns and 'surface' in df_feat.columns:
        df_feat['course_id'] = df_feat['place_code'].astype(str) + "_" + df_feat['surface'].astype(str) + "_" + df_feat['distance_num'].astype(str)
    else:
        df_feat['course_id'] = "default"

    # スタート指数の過去平均（脚質の代理指標）
    s_vals = pd.to_numeric(df_feat.get('my_start_idx', pd.Series()), errors='coerce')
    df_feat['eff_my_start_idx'] = s_vals.groupby(df_feat['馬名_clean']).apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    # 1. 距離適性特徴量
    dist_diffs = []
    cat_win_rates = []
    cat_runs_list = []

    for horse, group in df_feat.groupby('馬名_clean', sort=False):
        for i in range(len(group)):
            curr_row = group.iloc[i]
            past_rows = group.iloc[:i]

            curr_dist = curr_row['distance_num']
            curr_cat = curr_row['dist_cat']

            if past_rows.empty:
                dist_diffs.append(np.nan)
                cat_win_rates.append(np.nan)
                cat_runs_list.append(0)
            else:
                top3_past = past_rows[past_rows['is_top3_past'] == 1]
                if not top3_past.empty and pd.notna(curr_dist):
                    best_dist_avg = top3_past['distance_num'].mean()
                    dist_diffs.append(abs(curr_dist - best_dist_avg))
                else:
                    dist_diffs.append(np.nan)

                cat_past = past_rows[past_rows['dist_cat'] == curr_cat]
                c_runs = len(cat_past)
                cat_runs_list.append(c_runs)
                cat_win_rates.append(cat_past['is_win_past'].sum() / c_runs if c_runs > 0 else np.nan)

    df_feat['dist_diff'] = dist_diffs
    df_feat['cat_win_rate'] = cat_win_rates
    df_feat['cat_runs'] = cat_runs_list

    # 2. コース×脚質適合度（コースごとの先行好走率）
    course_front_rates = df_feat.groupby('course_id')['is_top3_past'].mean().to_dict()
    df_feat['course_front_rate'] = df_feat['course_id'].map(course_front_rates).fillna(0.3)
    # 脚質適合度 = 馬のスタート指数 × コースの前残り傾向
    df_feat['style_course_fit'] = df_feat['eff_my_start_idx'].fillna(50) * df_feat['course_front_rate']

    # レース間隔
    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days

    # 近走パフォーマンス
    target_cols = ['my_time_idx', 'my_last3f_idx', 'my_pace_idx']
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
    
    print("Processing features (Distance + Course/Style Fitness)...")
    df_prep = preprocess_features(df_clean)

    candidate_features = [
        '馬番', '枠番', '斤量', 'distance_num',
        'dist_diff', 'cat_win_rate', 'cat_runs',
        'course_front_rate', 'style_course_fit', # 新規追加（コース×脚質適合度）
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