import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

INPUT_CSV = "ml_target_data.csv"
MODEL_FILE = "keiba_ai_model.pkl"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ {INPUT_CSV} が見つかりません。先に create_features.py を実行してください。")
        return

    print(f"📂 事前計算済みの最強データ {INPUT_CSV} を読み込んでいます...")
    df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')

    # 性別・馬体重の処理
    if '性齢' in df.columns:
        df['sex_code'] = df['性齢'].astype(str).str[0].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)
        df['age'] = df['性齢'].astype(str).str[1:].apply(pd.to_numeric, errors='coerce')
    else:
        df['sex_code'], df['age'] = 0, np.nan

    if '馬体重' in df.columns:
        df['horse_weight'] = df['馬体重'].astype(str).str.extract(r'^(\d+)').astype(float)
        df['weight_change'] = df['馬体重'].astype(str).str.extract(r'\(([-+]?\d+)\)').astype(float)
    else:
        df['horse_weight'], df['weight_change'] = np.nan, np.nan

    # コース条件（芝ダート等）の数値化
    le_surf, le_cond = LabelEncoder(), LabelEncoder()
    if 'surface' in df.columns:
        df['surface_code'] = le_surf.fit_transform(df['surface'].fillna('不明').astype(str))
    else:
        df['surface_code'] = 0

    if 'condition' in df.columns:
        df['condition_code'] = le_cond.fit_transform(df['condition'].fillna('不明').astype(str))
    else:
        df['condition_code'] = 0

    # ==========================================
    # 🔥 AIに学習させるデータ項目（カンニング排除・自作指数追加）
    # ==========================================
    features = [
        '枠番', '馬番', '斤量', 'sex_code', 'age', 'horse_weight', 'weight_change',
        'surface_code', 'condition_code', 
        'horse_win_rate', 'prev_rank',           # 馬の実績
        'horse_avg_time_idx',                    # ← 自作タイム指数
        'horse_avg_last3f_idx',                  # ← 自作上がり指数
        'horse_avg_pace_idx',                    # ← 自作追走（ペース）指数
        'horse_avg_start_idx',                   # ← 自作スタート指数
        'jockey_win_rate', 'jockey_track_win_rate' # 騎手の手腕と適性
    ]
    
    # 距離データがあれば追加
    if 'distance' in df.columns:
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        features.append('distance')

    # X（入力データ）と y（目的変数：1着かどうか）を分ける
    X = df[features].copy().apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean()).fillna(0)  # 欠損値は全体の平均または0で埋める
    y = df['is_win']

    print(f"📊 学習データ件数: {len(X)} 件 (AIの脳みそを構築中...)")
    
    # 過学習を防ぐため 80%を学習用、20%をテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 最新のアルゴリズム（LightGBM）でトレーニングを開始します...")
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # モデルと使用した特徴量リスト、エンコーダーを丸ごと保存
    joblib.dump({
        'model': model, 
        'features': features, 
        'le_surf': le_surf, 
        'le_cond': le_cond
    }, MODEL_FILE)
    
    print(f"\n🎉 完了！指数の計算までマスターした完全体AIが {MODEL_FILE} に保存されました！")

if __name__ == "__main__":
    main()