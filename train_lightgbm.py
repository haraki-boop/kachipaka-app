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
    s = re.sub(r'[\s・･.\-ー_]+', '', s).strip()
    return s.upper()

def get_dist_cat(d):
    if pd.isna(d): return np.nan
    if d <= 1400: return 'sprint'
    elif d <= 2200: return 'mile_middle'
    else: return 'stayer'

def parse_sex_age(val):
    if pd.isna(val): return 0, 4.0
    val = str(val).strip()
    sex_char = val[0] if len(val) > 0 else '牡'
    sex_code = 0 if sex_char == '牡' else (1 if sex_char == '牝' else 2)
    try: age = float(val[1:])
    except: age = 4.0
    return sex_code, age

def parse_weight(val):
    if pd.isna(val): return np.nan, np.nan
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return np.nan, np.nan

def parse_passing(val):
    if pd.isna(val): return np.nan, np.nan, np.nan
    parts = str(val).split('-')
    try:
        f_pos = float(parts[0])
        l_pos = float(parts[-1])
        diff = f_pos - l_pos
        return f_pos, l_pos, diff
    except:
        return np.nan, np.nan, np.nan

def preprocess_features(df):
    df_feat = df.copy()

    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    df_feat = df_feat.dropna(subset=['date_parsed']).sort_values(['馬名_clean', 'date_parsed'])

    sex_age = df_feat['性齢'].apply(parse_sex_age)
    df_feat['sex_code'] = [s[0] for s in sex_age]
    df_feat['age'] = [s[1] for s in sex_age]

    df_feat['distance_num'] = pd.to_numeric(df_feat.get('distance'), errors='coerce')
    df_feat['dist_cat'] = df_feat['distance_num'].apply(get_dist_cat)
    df_feat['rank_num'] = pd.to_numeric(df_feat.get('着順'), errors='coerce')
    df_feat['is_top3_past'] = (df_feat['rank_num'] <= 3).astype(int)
    df_feat['is_win_past'] = (df_feat['rank_num'] == 1).astype(int)

    passing = df_feat['通過'].apply(parse_passing)
    df_feat['first_corner_pos'] = [p[0] for p in passing]
    df_feat['last_corner_pos'] = [p[1] for p in passing]
    df_feat['pos_gain'] = [p[2] for p in passing]

    df_feat['eff_rank_avg'] = df_feat.groupby('馬名_clean')['rank_num'].apply(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    
    df_feat['eff_top5_rate'] = df_feat.groupby('馬名_clean')['rank_num'].apply(
        lambda x: (x.shift(1) <= 5).astype(float).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df_feat['eff_top3_rate'] = df_feat.groupby('馬名_clean')['is_top3_past'].apply(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df_feat['kinryo_num'] = pd.to_numeric(df_feat.get('斤量'), errors='coerce')
    df_feat['wakuban_num'] = pd.to_numeric(df_feat.get('枠番'), errors='coerce')
    df_feat['umaban_num'] = pd.to_numeric(df_feat.get('馬番'), errors='coerce')

    weights_parsed = df_feat.get('馬体重', pd.Series()).apply(parse_weight)
    df_feat['body_weight'] = [p[0] for p in weights_parsed]
    df_feat['body_weight_diff'] = [p[1] for p in weights_parsed]
    df_feat['kinryo_body_ratio'] = df_feat['kinryo_num'] / df_feat['body_weight'].fillna(470)

    num_cols = [
        'horse_runs', 'horse_wins', 'horse_win_rate', 'prev_rank',
        'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx',
        'jockey_runs', 'jockey_wins', 'jockey_win_rate', 'jockey_track_win_rate',
        'jp_runs', 'jp_wins', 'first_pos'
    ]
    for c in num_cols:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors='coerce')

    df_feat['jp_win_rate'] = df_feat['jp_wins'] / (df_feat['jp_runs'] + 1e-5)

    place_code = df_feat.get('place_code', pd.Series(['00']*len(df_feat)))
    surface = df_feat.get('surface', pd.Series(['芝']*len(df_feat)))
    df_feat['course_id'] = place_code.astype(str) + "_" + surface.astype(str) + "_" + df_feat['distance_num'].fillna(0).astype(int).astype(str)
    df_feat['course_frame_id'] = df_feat['course_id'] + "_frame_" + df_feat['wakuban_num'].fillna(0).astype(int).astype(str)

    course_frame_win_rates = df_feat.groupby('course_frame_id')['is_win_past'].mean().to_dict()
    df_feat['course_frame_win_rate'] = df_feat['course_frame_id'].map(course_frame_win_rates).fillna(0.08)

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
                prev_kinryo = prev_row['kinryo_num']
                kinryo_diffs.append(curr_kinryo - prev_kinryo if pd.notna(curr_kinryo) and pd.notna(prev_kinryo) else 0.0)

                prev_jockey = str(prev_row.get('騎手', '')).strip()
                is_same_jockeys.append(1 if (curr_jockey and curr_jockey == prev_jockey) else 0)

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

                prev_prizes.append(pd.to_numeric(prev_row.get('賞金(万円)'), errors='coerce'))

    df_feat['kinryo_diff'] = kinryo_diffs
    df_feat['is_same_jockey'] = is_same_jockeys
    df_feat['dist_diff'] = dist_diffs
    df_feat['cat_win_rate'] = cat_win_rates
    df_feat['cat_runs'] = cat_runs_list
    df_feat['prev_prize'] = prev_prizes

    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days.fillna(30)
    df_feat['is_long_rest'] = (df_feat['interval_days'] >= 180).astype(int)

    target_cols = ['my_time_idx', 'my_last3f_idx', 'my_pace_idx', 'my_start_idx']
    for col in target_cols:
        if col in df_feat.columns:
            num_col = pd.to_numeric(df_feat[col], errors='coerce').clip(40, 80)
            df_feat[f'eff_{col}'] = num_col.groupby(df_feat['馬名_clean']).apply(
                lambda x: x.shift(1).rolling(3, min_periods=1).median()
            ).reset_index(level=0, drop=True)
        else:
            df_feat[f'eff_{col}'] = np.nan

    course_front_rates = df_feat.groupby('course_id')['is_top3_past'].mean().to_dict()
    df_feat['course_front_rate'] = df_feat['course_id'].map(course_front_rates).fillna(0.3)
    df_feat['style_course_fit'] = df_feat['eff_my_start_idx'].fillna(50) * df_feat['course_front_rate']

    df_feat['prev_rank_num'] = df_feat['rank_num'].groupby(df_feat['馬名_clean']).shift(1)

    # 差分指標（直近と通算の比較）
    df_feat['time_idx_diff'] = df_feat['eff_my_time_idx'] - df_feat['horse_avg_time_idx']
    df_feat['last3f_idx_diff'] = df_feat['eff_my_last3f_idx'] - df_feat['horse_avg_last3f_idx']
    df_feat['pace_idx_diff'] = df_feat['eff_my_pace_idx'] - df_feat['horse_avg_pace_idx']
    df_feat['start_idx_diff'] = df_feat['eff_my_start_idx'] - df_feat['horse_avg_start_idx']

    # Zスコア
    z_cols = [
        'eff_rank_avg', 'eff_top3_rate', 'prev_prize', 'eff_my_time_idx', 
        'eff_my_last3f_idx', 'eff_my_pace_idx', 'jockey_win_rate', 'jockey_track_win_rate',
        'horse_win_rate', 'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx',
        'age', 'kinryo_num', 'body_weight', 'prev_rank', 'first_pos'
    ]
    for c in z_cols:
        if c in df_feat.columns:
            df_feat[f'{c}_z'] = df_feat.groupby('race_id')[c].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-5) if len(x) > 1 else 0.0
            ).fillna(0.0)

    # メンバー内順位
    rank_cols = ['eff_my_time_idx', 'horse_avg_time_idx', 'jockey_win_rate', 'horse_win_rate']
    for c in rank_cols:
        if c in df_feat.columns:
            df_feat[f'{c}_rank_in_race'] = df_feat.groupby('race_id')[c].rank(ascending=False, method='min')

    return df_feat

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print("Loading data...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='cp932')

    df['rank_num_target'] = pd.to_numeric(df['着順'], errors='coerce')
    df_clean = df.dropna(subset=['rank_num_target', 'race_id']).copy()

    def calc_relevance(r):
        if r == 1: return 4
        elif r == 2: return 3
        elif r == 3: return 2
        elif r <= 5: return 1
        return 0

    df_clean['relevance'] = df_clean['rank_num_target'].apply(calc_relevance)

    print("Engineering 68 full features...")
    df_prep = preprocess_features(df_clean)
    df_prep = df_prep.sort_values(by='race_id').reset_index(drop=True)

    candidate_features = [
        # 🔥【強】グループ（20項目）
        'eff_my_time_idx_z', 'eff_rank_avg_z', 'eff_top3_rate_z', 'jockey_track_win_rate_z',
        'eff_my_time_idx_rank_in_race', 'horse_avg_time_idx_rank_in_race', 'jockey_win_rate_rank_in_race', 'horse_win_rate_rank_in_race',
        'horse_win_rate_z', 'horse_avg_time_idx_z', 'horse_avg_last3f_idx_z', 'horse_avg_pace_idx_z', 'horse_avg_start_idx_z',
        'prev_prize_z', 'prev_rank_z', 'first_pos_z',
        
        # ⚖️【中】グループ（22項目）
        'time_idx_diff', 'last3f_idx_diff', 'pace_idx_diff', 'start_idx_diff',
        'kinryo_body_ratio', 'body_weight_z', 'kinryo_diff', 'is_same_jockey', 'dist_diff',
        'cat_win_rate', 'course_front_rate', 'course_frame_win_rate', 'style_course_fit',
        'eff_my_last3f_idx_z', 'eff_my_pace_idx_z', 'eff_top5_rate', 'prev_rank_num',
        'first_corner_pos', 'last_corner_pos', 'pos_gain', 'jp_win_rate',
        
        # 🔹【小】グループ（26項目）
        '枠番', '馬番', 'sex_code', 'age', 'age_z', 'kinryo_num', 'body_weight', 'body_weight_diff',
        'distance_num', 'cat_runs', 'interval_days', 'is_long_rest', 'place_code',
        'eff_rank_avg', 'eff_my_time_idx', 'eff_my_last3f_idx', 'eff_my_pace_idx', 'eff_my_start_idx',
        'horse_runs', 'horse_wins', 'horse_win_rate', 'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx',
        'jockey_runs', 'jockey_wins', 'jockey_win_rate', 'jockey_track_win_rate', 'jp_runs', 'jp_wins'
    ]

    use_features = [f for f in candidate_features if f in df_prep.columns]

    X = df_prep[use_features].copy()
    y = df_prep['relevance']

    groups = df_prep.groupby('race_id', sort=False).size().values

    print(f"Training Full 68-Feature Lambdarank model on {len(groups)} races...")

    train_data = lgb.Dataset(X, label=y, group=groups)
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'eval_at': [1, 3],
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 18,
        'min_data_in_leaf': 40,
        'feature_fraction': 0.85,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, train_data, num_boost_round=200)
    joblib.dump({'model': model, 'features': use_features}, MODEL_FILE)
    print(f"🎉 Success! Model saved with {len(use_features)} features to {MODEL_FILE}")

if __name__ == "__main__":
    main()