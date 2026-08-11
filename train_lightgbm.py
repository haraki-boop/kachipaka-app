import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

INPUT_CSV = "ml_target_data.csv"
MODEL_FILE = "keiba_ai_model.pkl"

def preprocess_features(df):
    df_feat = df.copy()

    # 1. 日付と馬名でソート（時系列の担保）
    if 'date' in df_feat.columns and '馬名' in df_feat.columns:
        df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
        df_feat = df_feat.sort_values(['馬名', 'date_parsed'])

    # 2. オッズと人気の処理（市場データ）
    raw_odds = pd.to_numeric(df_feat.get('単勝', df_feat.get('オッズ', pd.Series())), errors='coerce').fillna(15.0)
    df_feat['log_odds'] = np.log(raw_odds.clip(lower=1.1))
    df_feat['pop_num'] = pd.to_numeric(df_feat.get('人気'), errors='coerce').fillna(99.0)

    # 3. 新ファクター：レース間隔（日数）
    if 'date_parsed' in df_feat.columns:
        df_feat['interval_days'] = df_feat.groupby('馬名')['date_parsed'].diff().dt.days.fillna(60.0)
    else:
        df_feat['interval_days'] = 60.0

    # 4. 新ファクター：前走の賞金（レースレベル・格）
    df_feat['prize_num'] = pd.to_numeric(df_feat.get('賞金(万円)'), errors='coerce').fillna(0.0)
    df_feat['prev_prize'] = df_feat.groupby('馬名')['prize_num'].shift(1).fillna(0.0)

    # 5. カンニング防止＆近走パフォーマンス（新ファクター：展開/脚質を統合）
    target_cols = ['my_time_idx', 'my_last3f_idx', 'my_pace_idx', 'my_start_idx']
    for col in target_cols:
        if col in df_feat.columns:
            num_col = pd.to_numeric(df_feat[col], errors='coerce')
            df_feat[f'recent3_{col}'] = num_col.groupby(df_feat['馬名']).apply(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            ).reset_index(level=0, drop=True)
        else:
            df_feat[f'recent3_{col}'] = np.nan

    df_feat['eff_time_idx'] = df_feat['recent3_my_time_idx'].fillna(75.0)
    df_feat['eff_last3f_idx'] = df_feat['recent3_my_last3f_idx'].fillna(75.0)
    df_feat['eff_pace_idx'] = df_feat['recent3_my_pace_idx'].fillna(75.0)
    df_feat['eff_start_idx'] = df_feat['recent3_my_start_idx'].fillna(85.0)
    
    # 6. 前走着順
    df_feat['prev_rank_num'] = pd.to_numeric(df_feat.get('prev_rank'), errors='coerce').fillna(9.0)

    # 7. 騎手・環境・馬実績
    j_col = df_feat.get('jockey_win_power', df_feat.get('jockey_win_rate', pd.Series()))
    df_feat['eff_jockey_win'] = pd.to_numeric(j_col, errors='coerce').fillna(0.05).clip(0.0, 1.0)
    df_feat['eff_jockey_track_win'] = pd.to_numeric(df_feat.get('jockey_track_win_rate'), errors='coerce').fillna(0.05).clip(0.0, 1.0)
    df_feat['horse_win_rate_val'] = pd.to_numeric(df_feat.get('horse_win_rate'), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    df_feat['horse_runs_val'] = pd.to_numeric(df_feat.get('horse_runs'), errors='coerce').fillna(0.0)
    df_feat['course_avg_time_val'] = pd.to_numeric(df_feat.get('course_avg_time'), errors='coerce').fillna(100.0)

    # 8. レース内偏差値（z-score）の算出
    if 'race_id' in df_feat.columns:
        df_feat['z_time_idx'] = df_feat.groupby('race_id')['eff_time_idx'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
        df_feat['z_last3f_idx'] = df_feat.groupby('race_id')['eff_last3f_idx'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
        df_feat['z_odds'] = df_feat.groupby('race_id')['log_odds'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))
    else:
        df_feat['z_time_idx'] = 0.0
        df_feat['z_last3f_idx'] = 0.0
        df_feat['z_odds'] = 0.0

    return df_feat

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='cp932')

    if 'is_win' not in df.columns:
        if '着順' in df.columns:
            df['is_win'] = (pd.to_numeric(df['着順'], errors='coerce') == 1).astype(int)
        else:
            print("Error: Target variable not found.")
            return

    df_clean = df.dropna(subset=['is_win']).copy()
    df_prep = preprocess_features(df_clean)

    # 追加ファクターを含んだ統合リスト
    candidate_features = [
        '馬番', '枠番', '斤量', 'distance',
        'log_odds', 'pop_num', 'z_odds',
        'interval_days', 'prev_prize', 
        'prev_rank_num', 'horse_win_rate_val', 'horse_runs_val',
        'eff_time_idx', 'eff_last3f_idx', 'eff_pace_idx', 'eff_start_idx',
        'z_time_idx', 'z_last3f_idx',
        'eff_jockey_win', 'eff_jockey_track_win',
        'course_avg_time_val'
    ]

    if 'surface_code' in df_prep.columns: candidate_features.append('surface_code')
    if 'condition_code' in df_prep.columns: candidate_features.append('condition_code')
    if 'sex_code' in df_prep.columns: candidate_features.append('sex_code')

    use_features = [f for f in candidate_features if f in df_prep.columns]
    print(f"Selected Features ({len(use_features)}): {use_features}")

    X = df_prep[use_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    y = df_prep['is_win']

    print(f"Training LightGBM model with {len(X)} records...")
    
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
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()