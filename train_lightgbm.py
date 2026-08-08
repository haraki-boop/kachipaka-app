import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

DB_CSV = "enhanced_keiba_data.csv"
MODEL_FILE = "keiba_ai_model.pkl"

def main():
    print(f"📂 完全体データベース {DB_CSV} を読み込んでいます...")
    try:
        df = pd.read_csv(DB_CSV, low_memory=False, encoding='utf-8-sig', dtype=str)
    except Exception:
        df = pd.read_csv(DB_CSV, low_memory=False, encoding='cp932', dtype=str)
        
    print("🛠️ データの前処理（コース適性と騎手手腕を含む能力評価）を開始します...")
    
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df = df.dropna(subset=['着順'])
    df['is_win'] = (df['着順'] == 1).astype(int)
    
    def time_to_sec(t):
        if pd.isna(t): return np.nan
        parts = str(t).strip().split(':')
        if len(parts) == 2: return float(parts[0]) * 60 + float(parts[1])
        try: return float(parts[0])
        except: return np.nan
        
    if 'タイム' in df.columns:
        df['time_seconds'] = df['タイム'].apply(time_to_sec)
    else:
        df['time_seconds'] = np.nan
        
    if '馬体重' in df.columns:
        df['horse_weight'] = df['馬体重'].astype(str).str.extract(r'^(\d+)').astype(float)
        df['weight_change'] = df['馬体重'].astype(str).str.extract(r'\(([-+]?\d+)\)').astype(float)
    else:
        df['horse_weight'] = np.nan
        df['weight_change'] = np.nan

    if '性齢' in df.columns:
        df['sex_code'] = df['性齢'].astype(str).str[0].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        df['age'] = df['性齢'].astype(str).str[1:].apply(pd.to_numeric, errors='coerce')
    else:
        df['sex_code'] = 0
        df['age'] = np.nan
        
    for col in ['枠番', '馬番', '斤量']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ==========================================
    # 🔥 コース適性と騎手データを復活・新規追加 🔥
    # ==========================================
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    
    le_surf = LabelEncoder()
    df['surface_code'] = le_surf.fit_transform(df['surface'].fillna('不明').astype(str))
    
    le_cond = LabelEncoder()
    df['condition_code'] = le_cond.fit_transform(df['condition'].fillna('不明').astype(str))
    
    # 騎手の全体勝率を計算
    jockey_stats = df.groupby('騎手')['is_win'].mean().reset_index()
    jockey_stats.rename(columns={'is_win': 'jockey_win_rate'}, inplace=True)
    df = pd.merge(df, jockey_stats, on='騎手', how='left')
    
    # 騎手の「競馬場別」勝率を計算（得意・不得意の判別）
    if 'place_name' not in df.columns and 'place' in df.columns:
        df['place_name'] = df['place']
        
    if 'place_name' in df.columns:
        jockey_place_stats = df.groupby(['騎手', 'place_name'])['is_win'].mean().reset_index()
        jockey_place_stats.rename(columns={'is_win': 'jockey_track_win_rate'}, inplace=True)
        df = pd.merge(df, jockey_place_stats, on=['騎手', 'place_name'], how='left')
    else:
        df['jockey_track_win_rate'] = df['jockey_win_rate']
    # ==========================================

    features = [
        '枠番', '馬番', '斤量', 'time_seconds', 'sex_code', 'age', 
        'horse_weight', 'weight_change', 'distance', 'surface_code', 'condition_code',
        'jockey_win_rate', 'jockey_track_win_rate'
    ]
    
    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    
    X = X.fillna(X.mean())
    if 'time_seconds' in X.columns:
        X['time_seconds'] = X['time_seconds'].fillna(100.0)
    X = X.fillna(0)
    
    y = df['is_win'].copy()
    
    print(f"📊 学習データ件数: {len(X)} 件")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 LightGBMモデルをトレーニング中... (コース適性・騎手手腕を考慮)")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    joblib.dump({
        'model': model,
        'features': features,
        'le_surf': le_surf,
        'le_cond': le_cond
    }, MODEL_FILE)
    
    print(f"\n🎉 学習完了！最強のAIモデルを {MODEL_FILE} として保存しました！")

if __name__ == "__main__":
    main()