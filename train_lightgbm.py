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

def parse_weight(val):
    """馬体重文字列（例: 480(+4)）から馬体重と増減を抽出"""
    if pd.isna(val): return np.nan, np.nan
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return np.nan, np.nan

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

    # 数値変換
    df_feat['kinryo_num'] = pd.to_numeric(df_feat.get('斤量'), errors='coerce')
    df_feat['wakuban_num'] = pd.to_numeric(df_feat.get('枠番'), errors='coerce')
    df_feat['umaban_num'] = pd.to_numeric(df_feat.get('馬番'), errors='coerce')

    # 馬体重と増減のパース
    weights_parsed = df_feat.get('馬体重', pd.Series()).apply(parse_weight)
    df_feat['body_weight'] = [p[0] for p in weights_parsed]
    df_feat['body_weight_diff'] = [p[1] for p in weights_parsed]
    # 斤量比（斤量 / 馬体重）
    df_feat['kinryo_body_ratio'] = df_feat['kinryo_num'] / df_feat['body_weight']

    # コースIDの作成（例: 札幌_芝_1800m）
    if 'place_code' in df_feat.columns and 'surface' in df_feat.columns:
        df_feat['course_id'] = df_feat['place_code'].astype(str) + "_" + df_feat['surface'].astype(str) + "_" + df_feat['distance_num'].astype(str)
    else:
        df_feat['course_id'] = "default"

    # コース×枠順勝率マップ（コースごとの枠順勝率）
    df_feat['course_frame_id'] = df_feat['course_id'] + "_frame_" + df_feat['wakuban_num'].fillna(0).astype(int).astype(str)
    course_frame_win_rates = df_feat.groupby('course_frame_id')['is_win_past'].mean().to_dict()
    df_feat['course_frame_win_rate'] = df_feat['course_frame_id'].map(course_frame_win_rates).fillna(0.08)

    # 時系列処理（馬ごとの過去比較）
    kinryo_diffs = []
    is_same_jockeys = []
    dist_diffs = []
    cat_win_rates = []
    cat_runs_list = []
    prev_prizes = []

    for horse, group in df_feat.groupby('馬名_clean', sort=False):
        for i in range(len(group)):
            curr_row = group.iloc[i]
            past_rows = group.iloc[:i]

            curr_dist = curr_row['distance_num']
            curr_cat = curr_row['dist_cat']
            curr_jockey = str(curr_row.get('騎手', '')).strip()
            curr_kinryo = curr_row['kinryo_num']

            if past_rows.empty:
                kinryo_diffs.append(0.0)
                is_same_jockeys.append(0)
                dist_diffs.append(np.nan)
                cat_win_rates.append(np.nan)
                cat_runs_list.append(0)
                prev_prizes.append(np.nan)
            else:
                prev_row = past_rows.iloc[-1]
                
                # 1. 斤量差（今回 - 前走）
                prev_kinryo = prev_row['kinryo_num']
                kinryo_diffs.append(curr_kinryo - prev_kinryo if pd.notna(curr_kinryo) and pd.notna(prev_kinryo) else 0.0)

                # 2. 継続騎乗フラグ（前走と同じ騎手か）
                prev_jockey = str(prev_row.get('騎手', '')).strip()
                is_same_jockeys.append(1 if (curr_jockey and curr_jockey == prev_jockey) else 0)

                # 3. 距離差（好走平均との差）
                top3_past = past_rows[past_rows['is_top3_past'] == 1]
                if not top3_past.empty and pd.notna(curr_dist):
                    best_dist_avg = top3_past['distance_num'].mean()
                    dist_diffs.append(abs(curr_dist - best_dist_avg))
                else:
                    dist_diffs.append(np.nan)

                # 4. 同距離カテゴリ勝率
                cat_past = past_rows[past_rows['dist_cat'] == curr_cat]
                c_runs = len(cat_past)
                cat_runs_list.append(c_runs)
                cat_win_rates.append(cat_past['is_win_past'].sum() / c_runs if c_runs > 0 else np.nan)

                # 5. 前走賞金（クラス格差の代理指標）
                prev_prizes.append(pd.to_numeric(prev_row.get('賞金(万円)'), errors='coerce'))

    df_feat['kinryo_diff'] = kinryo_diffs
    df_feat['is_same_jockey'] = is_same_jockeys
    df_feat['dist_diff'] = dist_diffs
    df_feat['cat_win_rate'] = cat_win_rates
    df_feat['cat_runs'] = cat_runs_list
    df_feat['prev_prize'] = prev_prizes

    # コース×脚質適合度
    s_vals = pd.to_numeric(df_feat.get('my_start_idx', pd.Series()), errors='coerce')
    df_feat['eff_my_start_idx'] = s_vals.groupby(df_feat['馬名_clean']).apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    course_front_rates = df_feat.groupby('course_id')['is_top3_past'].mean().to_dict()
    df_feat['course_front_rate'] = df_feat['course_id'].map(course_front_rates).fillna(0.3)
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
    
    print("Processing ALL features (Distance + Course/Frame + Kinryo + Jockey + Weight + Class)...")
    df_prep = preprocess_features(df_clean)

    candidate_features = [
        'umaban_num', 'wakuban_num', 'kinryo_num', 'distance_num',
        'kinryo_diff', 'kinryo_body_ratio', 'body_weight_diff', # 斤量・馬体重系
        'is_same_jockey', 'course_frame_win_rate', # 騎手継続・枠順適性
        'dist_diff', 'cat_win_rate', 'cat_runs', # 距離適性
        'course_front_rate', 'style_course_fit', # コース脚質適合度
        'interval_days', 'prev_prize', 'prev_rank_num', # クラス・展開系
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

    print(f"Training LightGBM model for WIN probability with {len(X)} records and {len(use_features)} features...")
    
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