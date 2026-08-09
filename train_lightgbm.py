import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

INPUT_CSV = "ml_target_data.csv"
MODEL_FILE = "keiba_ai_model.pkl"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='cp932')

    if '着順' not in df.columns:
        print("Error: Target variable '着順' not found.")
        return

    # オッズと人気を学習項目から完全排除
    features = [
        '馬番', '斤量', 'my_time_idx', 'my_last3f_idx', 
        'my_pace_idx', 'my_start_idx', 'jockey_win_power'
    ]
    if 'surface_code' in df.columns: features.append('surface_code')
    if 'condition_code' in df.columns: features.append('condition_code')
    if 'sex_code' in df.columns: features.append('sex_code')

    use_features = [f for f in features if f in df.columns]
    print(f"Selected Features: {use_features}")

    df_clean = df.dropna(subset=use_features + ['着順']).copy()
    X = df_clean[use_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df_clean['着順'] # 着順予測(Regression)に変更

    if len(X) < 100:
        print("Error: Not enough data for training.")
        return

    print(f"Training LightGBM model with {len(X)} records...")
    
    train_data = lgb.Dataset(X, label=y)
    params = {
        'objective': 'regression', # 回帰問題
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=150
    )

    save_data = {'model': model, 'features': use_features}
    
    if os.path.exists("le_surf.pkl"): save_data['le_surf'] = joblib.load("le_surf.pkl")
    if os.path.exists("le_cond.pkl"): save_data['le_cond'] = joblib.load("le_cond.pkl")

    joblib.dump(save_data, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()