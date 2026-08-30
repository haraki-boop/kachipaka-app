import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from catboost import CatBoost

MODEL_FILE = "keiba_ai_model.pkl"
TARGET_CSV = "ml_target_data_v2.csv"

# 🌟 pklファイルのロードに必要なアンサンブルクラス定義
class EnsembleModel:
    def __init__(self, lgb_model, xgb_model, cat_model, weights=(0.4, 0.3, 0.3)):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        X_num = X.copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)

        # 🌟 Zスコア化を削除し、回帰予測値（確率）をそのまま加重平均
        lgb_pred = self.lgb_model.predict(X_num)
        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_num))
        cat_pred = self.cat_model.predict(X_num)

        w1, w2, w3 = self.weights
        return w1 * lgb_pred + w2 * xgb_pred + w3 * cat_pred

def verify():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(TARGET_CSV):
        print("❌ モデルファイルまたはCSVファイルが見つかりません。")
        return

    print("📊 複勝圏内特化・特徴量選別モデルの検証（バックテスト）を開始します...")
    
    # 1. モデルとデータの読み込み
    model_data = joblib.load(MODEL_FILE)
    model = model_data['model']
    features = model_data['features']
    
    try:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='utf-8-sig')
    except:
        df = pd.read_csv(TARGET_CSV, low_memory=False, encoding='cp932')
    
    # 前処理
    df['rank_num'] = pd.to_numeric(df['着順'], errors='coerce')
    odds_col = '単勝' if '単勝' in df.columns else ('オッズ' if 'オッズ' in df.columns else None)
    if not odds_col:
        print("❌ オッズ列が見つかりません。")
        return
    df['odds_num'] = pd.to_numeric(df[odds_col].astype(str).str.replace('倍', ''), errors='coerce')
    
    # テスト期間（2026年6月〜8月）のデータ抽出
    df['date_norm'] = df['date'].astype(str).str.replace(r'\D', '', regex=True).str[:8]
    test_df = df[df['date_norm'] >= '20260601'].dropna(subset=['rank_num', 'odds_num', 'race_id']).copy()
    
    if test_df.empty:
        test_df = df.dropna(subset=['rank_num', 'odds_num', 'race_id']).tail(int(len(df)*0.2)).copy()

    # 特徴量の前処理（地方ロジックで追加された特徴量を生成）
    def parse_passing(val):
        if pd.isna(val): return np.nan, np.nan, np.nan
        parts = str(val).split('-')
        try: return float(parts[0]), float(parts[-1]), float(parts[0]) - float(parts[-1])
        except: return np.nan, np.nan, np.nan

    def parse_weight(val):
        import re
        if pd.isna(val): return np.nan, np.nan
        s = str(val).strip()
        m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
        if m: return float(m.group(1)), float(m.group(2)) if m.group(2) else 0.0
        return np.nan, np.nan

    def clean_horse_name(name):
        import unicodedata, re
        if pd.isna(name): return ""
        s = unicodedata.normalize('NFKC', str(name))
        s = re.sub(r'[\s・･.\-ー_]+', '', s).strip()
        return s.upper()

    test_df['馬名_clean'] = test_df['馬名'].astype(str).apply(clean_horse_name)
    test_df['date_parsed'] = pd.to_datetime(test_df['date'], errors='coerce')
    test_df['distance_num'] = pd.to_numeric(test_df.get('distance'), errors='coerce')
    test_df['place_code'] = test_df.get('place_code', pd.Series(['00']*len(test_df))).astype(str)
    
    def get_dist_cat(d):
        if pd.isna(d): return np.nan
        if d <= 1400: return 'sprint'
        elif d <= 2200: return 'mile_middle'
        else: return 'stayer'
    test_df['dist_cat'] = test_df['distance_num'].apply(get_dist_cat)

    test_df['prev_dist'] = test_df.groupby('馬名_clean')['distance_num'].shift(1)
    test_df['dist_change_num'] = test_df['distance_num'] - test_df['prev_dist'].fillna(test_df['distance_num'])
    
    test_df['same_dist_avg_rank'] = test_df.groupby(['馬名_clean', 'dist_cat'])['rank_num'].apply(
        lambda x: x.shift(1).expanding().mean()
    ).reset_index(level=[0,1], drop=True).fillna(7.0)

    test_df['same_place_avg_rank'] = test_df.groupby(['馬名_clean', 'place_code'])['rank_num'].apply(
        lambda x: x.shift(1).expanding().mean()
    ).reset_index(level=[0,1], drop=True).fillna(7.0)

    prize_col_name = '賞金(万円)' if '賞金(万円)' in test_df.columns else 'prize'
    test_df['prize_num'] = pd.to_numeric(test_df.get(prize_col_name, 0), errors='coerce').fillna(0.0)
    test_df['horse_prize_avg'] = test_df.groupby('馬名_clean')['prize_num'].apply(
        lambda x: x.shift(1).expanding().mean()
    ).reset_index(level=0, drop=True).fillna(0.0)
    
    test_df['race_avg_prize'] = test_df.groupby('race_id')['horse_prize_avg'].transform('mean').replace(0, 1)
    test_df['race_prize_relative'] = test_df['horse_prize_avg'] / test_df['race_avg_prize']
    test_df['race_prize_rank'] = test_df.groupby('race_id')['horse_prize_avg'].rank(ascending=False, method='min')

    passing = test_df['通過'].apply(parse_passing)
    test_df['first_corner'] = [p[0] for p in passing]
    test_df['last_corner'] = [p[1] for p in passing]
    test_df['corner_diff'] = [p[2] for p in passing]
    test_df['prev_1c'] = test_df.groupby('馬名_clean')['first_corner'].shift(1).fillna(10.0)

    # 🌟 追加: テンの安定度
    if 'my_start_idx' in test_df.columns:
        start_num = pd.to_numeric(test_df['my_start_idx'], errors='coerce')
        test_df['start_idx_avg'] = test_df.groupby('馬名_clean')[start_num.name].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True).fillna(50.0)
        test_df['start_idx_std'] = test_df.groupby('馬名_clean')[start_num.name].apply(lambda x: x.shift(1).rolling(3, min_periods=1).std()).reset_index(level=0, drop=True).fillna(0.0)
    else:
        test_df['start_idx_avg'] = 50.0
        test_df['start_idx_std'] = 0.0

    # 🌟 追加: 上がりの安定度
    if 'my_last3f_idx' in test_df.columns:
        last3f_num = pd.to_numeric(test_df['my_last3f_idx'], errors='coerce')
        test_df['last_3f_avg_rank'] = test_df.groupby('馬名_clean')[last3f_num.name].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True).fillna(50.0)
        test_df['last_3f_std'] = test_df.groupby('馬名_clean')[last3f_num.name].apply(lambda x: x.shift(1).rolling(3, min_periods=1).std()).reset_index(level=0, drop=True).fillna(0.0)
    else:
        test_df['last_3f_avg_rank'] = 50.0
        test_df['last_3f_std'] = 0.0

    test_df['kinryo_num'] = pd.to_numeric(test_df.get('斤量'), errors='coerce').fillna(55.0)
    weights_parsed = test_df.get('馬体重', pd.Series()).apply(parse_weight)
    test_df['body_weight'] = [p[0] for p in weights_parsed]
    test_df['kinryo_weight_ratio'] = test_df['kinryo_num'] / test_df['body_weight'].fillna(470)
    test_df['interval_days'] = test_df.groupby('馬名_clean')['date_parsed'].diff().dt.days.fillna(30)
    
    for c in ['horse_runs', 'jockey_win_rate', 'trainer_win_rate']:
        test_df[c] = pd.to_numeric(test_df.get(c, 0), errors='coerce').fillna(0.0)

    # 2. 予測実行
    X_test = pd.DataFrame(index=test_df.index)
    for f in features:
        X_test[f] = pd.to_numeric(test_df[f], errors='coerce').fillna(0) if f in test_df.columns else 0.0

    test_df['score'] = model.predict(X_test)
    test_df['ai_rank'] = test_df.groupby('race_id')['score'].rank(ascending=False, method='first')
    
    # 3. 成績集計
    total_races = test_df['race_id'].nunique()
    
    ai_top1 = test_df[test_df['ai_rank'] == 1]
    top1_wins = ai_top1[ai_top1['rank_num'] == 1]
    top1_top3s = ai_top1[ai_top1['rank_num'] <= 3]
    
    top1_win_rate = (len(top1_wins) / total_races) * 100 if total_races > 0 else 0
    top1_top3_rate = (len(top1_top3s) / total_races) * 100 if total_races > 0 else 0
    top1_return = (top1_wins['odds_num'].sum() / total_races) * 100 if total_races > 0 else 0
    
    ai_top3_horses = test_df[test_df['ai_rank'] <= 3]
    hit_races_3renpuku = 0
    for race_id, group in ai_top3_horses.groupby('race_id'):
        if len(group) == 3 and all(group['rank_num'] <= 3):
            hit_races_3renpuku += 1
            
    sanrenpuku_hit_rate = (hit_races_3renpuku / total_races) * 100 if total_races > 0 else 0

    # 4. 結果表示
    print("\n" + "="*55)
    print(f"🏁 複勝圏内特化・特徴量選別モデル 検証レポート（{total_races} レース）")
    print("="*55)
    print(f"【👑 本命（スコア1位）の成績】")
    print(f"  ・1着 的中率（勝率） : {top1_win_rate:.1f}%")
    print(f"  ・3着内 的中率(複勝率): {top1_top3_rate:.1f}%")
    print(f"  ・単勝 ベタ買い回収率 : {top1_return:.1f}%")
    print("-" * 55)
    print(f"【💎 上位3頭BOXの成績】")
    print(f"  ・（3連複 1点買い 的中率）: {sanrenpuku_hit_rate:.2f}%")
    print("="*55)

if __name__ == "__main__":
    verify()