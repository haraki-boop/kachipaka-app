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
    """馬名の正規化（全角半角・空白・記号の完全除去）"""
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip()

def preprocess_features(df):
    """学習用の特徴量エンジニアリング（嘘の穴埋めは一切しない）"""
    df_feat = df.copy()

    # 馬名の正規化と日付ソート
    df_feat['馬名_clean'] = df_feat['馬名'].astype(str).apply(clean_horse_name)
    df_feat['date_parsed'] = pd.to_datetime(df_feat['date'], errors='coerce')
    df_feat = df_feat.dropna(subset=['date_parsed']).sort_values(['馬名_clean', 'date_parsed'])

    # レース間隔（日数）
    df_feat['interval_days'] = df_feat.groupby('馬名_clean')['date_parsed'].diff().dt.days

    # 近走パフォーマンス（過去3走の平均を算出、欠損はNaNのまま）
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
    df_feat['prev_rank_num'] = pd.to_numeric(df_feat.get('prev_rank', df_feat.get('着順')), errors='coerce').groupby(df_feat['馬名_clean']).shift(1)

    # 各種実績データ
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

    # 目的変数：3着以内に入る確率（複勝率）
    df['is_top3'] = (pd.to_numeric(df['着順'], errors='coerce') <= 3).astype(int)
    df_clean = df.dropna(subset=['is_top3']).copy()
    
    print("Processing features...")
    df_prep = preprocess_features(df_clean)

    # 実際にモデルに食わせる特徴量のリスト
    candidate_features = [
        '馬番', '枠番', '斤量', 'distance',
        'interval_days', 'prev_prize', 'prev_rank_num', 
        'horse_win_rate_val', 'horse_runs_val',
        'eff_my_time_idx', 'eff_my_last3f_idx', 'eff_my_pace_idx', 'eff_my_start_idx',
        'eff_jockey_win', 'eff_jockey_track_win'
    ]

    use_features = [f for f in candidate_features if f in df_prep.columns]
    
    # カテゴリ変数の処理
    cat_cols = ['surface_code', 'condition_code', 'sex_code']
    for cat in cat_cols:
        if cat in df_prep.columns:
            df_prep[cat] = df_prep[cat].astype('category')
            use_features.append(cat)

    # X(特徴量)と y(正解ラベル)の定義
    X = df_prep[use_features].copy()
    y = df_prep['is_top3']

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
    
    # 欠損値(NaN)はLightGBMが自動で最適に分岐処理します
    model = lgb.train(params, train_data, num_boost_round=150)
    
    joblib.dump({'model': model, 'features': use_features}, MODEL_FILE)
    print(f"Success! Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()