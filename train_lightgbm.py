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
        
    print("🛠️ データの前処理（AI用の頭脳づくり）を開始します...")
    
    # 1. 目的変数（1着を予測）
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df = df.dropna(subset=['着順'])
    df['is_win'] = (df['着順'] == 1).astype(int)
    
    # 2. タイムと上がり3F
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
        
    if '上り' in df.columns:
        df['last_3f'] = pd.to_numeric(df['上り'], errors='coerce')
    else:
        df['last_3f'] = np.nan
        
    # 3. 馬体重・増減
    if '馬体重' in df.columns:
        df['horse_weight'] = df['馬体重'].astype(str).str.extract(r'^(\d+)').astype(float)
        df['weight_change'] = df['馬体重'].astype(str).str.extract(r'\(([-+]?\d+)\)').astype(float)
    else:
        df['horse_weight'] = np.nan
        df['weight_change'] = np.nan

    # 4. 性齢
    if '性齢' in df.columns:
        df['sex_code'] = df['性齢'].astype(str).str[0].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        df['age'] = df['性齢'].astype(str).str[1:].apply(pd.to_numeric, errors='coerce')
    else:
        df['sex_code'] = 0
        df['age'] = np.nan
        
    # 5. 脚質（通過順位から「逃げ・差し」などを数値化）
    def calc_position(p):
        if pd.isna(p): return np.nan
        nums = re.findall(r'\d+', str(p))
        if not nums: return np.nan
        return np.mean([int(n) for n in nums]) # 数値が小さいほど「逃げ・先行」
        
    if '通過' in df.columns:
        df['avg_position'] = df['通過'].apply(calc_position)
    else:
        df['avg_position'] = np.nan

    # 6. コース環境（芝ダート、距離、馬場状態）
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    
    # LabelEncoderの保存用（後でアプリ側で使えるようにします）
    le_surf = LabelEncoder()
    df['surface_code'] = le_surf.fit_transform(df['surface'].fillna('不明').astype(str))
    
    le_cond = LabelEncoder()
    df['condition_code'] = le_cond.fit_transform(df['condition'].fillna('不明').astype(str))
    
    # 7. その他の基本データ数値化
    for col in ['枠番', '馬番', '斤量']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 🎯 AIに与える最終的な「特徴量（考える材料）」リスト
    features = [
        '枠番', '馬番', '斤量', 'sex_code', 'age', 
        'horse_weight', 'weight_change',
        'time_seconds', 'last_3f', 'avg_position', 
        'distance', 'surface_code', 'condition_code'
    ]
    
    # 特徴量と正解データのセット
    X = df[features].copy()
    y = df['is_win'].copy()
    
    print(f"📊 学習データ件数: {len(X)} 件")
    
    # 学習用とテスト用（答え合わせ用）にデータを分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 LightGBMモデルをトレーニング中... (プロ馬券師の思考を注入中)")
    
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
    
    # 学習済みモデルと、エンコーダー（文字を数字に変換する辞書）を一緒に保存
    joblib.dump({
        'model': model,
        'le_surf': le_surf,
        'le_cond': le_cond,
        'features': features
    }, MODEL_FILE)
    
    print(f"\n🎉 学習完了！最強のAIモデルを {MODEL_FILE} として保存しました！")

if __name__ == "__main__":
    main()