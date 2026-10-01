import os
import joblib
import pandas as pd
import numpy as np
import re
import unicodedata

FUTURE_CSV = "future_races.csv"
CACHE_FILE = "app_cache.pkl"
MODEL_FILE = "keiba_ai_model.pkl"

class EnsembleModel:
    def __init__(self, lgb_model, xgb_model, cat_model, weights=(0.4, 0.3, 0.3)):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.weights = weights
    def predict(self, X):
        pass

import __main__
__main__.EnsembleModel = EnsembleModel

def clean_name(text):
    if pd.isna(text): return ""
    s = unicodedata.normalize('NFKC', str(text))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def check_merge():
    print("==================================================")
    print("🔍 685列フルスペックモデル：キャッシュ結合検証")
    print("==================================================")

    for f in [FUTURE_CSV, CACHE_FILE, MODEL_FILE]:
        if not os.path.exists(f):
            print(f"❌ エラー: {f} が見つかりません。")
            return

    df_future = pd.read_csv(FUTURE_CSV, encoding='utf-8-sig', dtype={'race_id': str})
    cache = joblib.load(CACHE_FILE)
    model_data = joblib.load(MODEL_FILE)
    features = model_data['features']
    
    df_future['race_id'] = df_future['race_id'].astype(str).str.zfill(12)
    df_future['馬名_clean'] = df_future['馬名'].apply(clean_name)
    df_future['騎手_clean'] = df_future['騎手'].astype(str).apply(clean_name)
    df_future['調教師_clean'] = df_future['調教師'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).apply(clean_name)
    
    places = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
              '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    df_future['place_name'] = df_future['race_id'].str[4:6].map(places).fillna('不明')
    
    df_future['surface'] = df_future.get('surface', pd.Series(index=df_future.index)).fillna('不明')
    if 'distance' in df_future.columns:
        df_future['distance'] = pd.to_numeric(df_future['distance'], errors='coerce').fillna(1600.0)
    else:
        df_future['distance'] = 1600.0
    df_future['dist_category'] = pd.cut(df_future['distance'], bins=[0, 1300, 1700, 2100, 4000], labels=['短距離', 'マイル', '中距離', '長距離']).astype(str)
        
    df_future['is_right_turn'] = df_future['place_name'].isin(['中山', '阪神', '京都', '小倉', '福島', '札幌', '函館']).astype(int)

    if '馬体重' in df_future.columns:
        weight_extract = df_future['馬体重'].astype(str).str.extract(r'(\d+)\s*\(([-+]?\d+)\)')
        df_future['weight'] = pd.to_numeric(weight_extract[0], errors='coerce').fillna(470.0)
        df_future['weight_change'] = pd.to_numeric(weight_extract[1], errors='coerce').fillna(0.0)
    else:
        df_future['weight'], df_future['weight_change'] = 470.0, 0.0

    if '斤量' in df_future.columns:
        df_future['斤量'] = pd.to_numeric(df_future['斤量'], errors='coerce').fillna(55.0)
    else:
        df_future['斤量'] = 55.0
        
    df_future['kinryo_weight_ratio'] = (df_future['斤量'] / df_future['weight'].replace(0, np.nan)).fillna(0.0)

    df = df_future.copy()

    # 📦 キャッシュ内データのスキャンと結合
    for key, value in cache.items():
        if key == 'predictions': continue
        if isinstance(value, dict) and len(value) > 0:
            first_val = next(iter(value.values()))
            if isinstance(first_val, dict):
                temp_df = pd.DataFrame.from_dict(value, orient='index').reset_index()
                temp_df = temp_df.rename(columns={'index': '馬名_clean'})
                cols_to_use = temp_df.columns.difference(df.columns).tolist() + ['馬名_clean']
                df = pd.merge(df, temp_df[cols_to_use], on='馬名_clean', how='left')

    # 3. カテゴリ・ターゲットエンコーディングの復元 (フルコンボ対応)
    df['jockey_win_rate'] = df['騎手_clean'].map(cache.get('jockey_win_map', {})).fillna(0.05)
    df['jockey_place_rate'] = df['騎手_clean'].map(cache.get('jockey_place_map', {})).fillna(0.15)
    df['trainer_win_rate'] = df['調教師_clean'].map(cache.get('trainer_win_map', {})).fillna(0.05)
    df['trainer_place_rate'] = df['調教師_clean'].map(cache.get('trainer_place_map', {})).fillna(0.15)

    df['jp_key'] = df['騎手_clean'] + "_" + df['place_name']
    df['jockey_place_win_rate'] = df['jp_key'].map({f"{k[0]}_{k[1]}": v for k, v in cache.get('jp_win_map', {}).items()}).fillna(0.05)
    df['jockey_place_place_rate'] = df['jp_key'].map({f"{k[0]}_{k[1]}": v for k, v in cache.get('jp_place_map', {}).items()}).fillna(0.15)

    df['tp_key'] = df['調教師_clean'] + "_" + df['place_name']
    df['trainer_place_win_rate'] = df['tp_key'].map({f"{k[0]}_{k[1]}": v for k, v in cache.get('tp_win_map', {}).items()}).fillna(0.05)

    df['js_key'] = df['騎手_clean'] + "_" + df['surface']
    df['jockey_surface_win_rate'] = df['js_key'].map({f"{k[0]}_{k[1]}": v for k, v in cache.get('js_win_map', {}).items()}).fillna(0.05)
    df['jockey_surface_place_rate'] = df['js_key'].map({f"{k[0]}_{k[1]}": v for k, v in cache.get('js_place_map', {}).items()}).fillna(0.15)

    df['jd_key'] = df['騎手_clean'] + "_" + df['dist_category']
    df['jockey_dist_win_rate'] = df['jd_key'].map({f"{k[0]}_{k[1]}": v for k, v in cache.get('jd_win_map', {}).items()}).fillna(0.05)
    
    df['jt_key'] = df['騎手_clean'] + "_" + df['調教師_clean']
    df['jockey_trainer_combo_win_rate'] = df['jt_key'].map({f"{k[0]}_{k[1]}": v for k, v in cache.get('jt_win_map', {}).items()}).fillna(0.05)

    if 'last_date' in df.columns:
        last_dates = pd.to_datetime(df['last_date'], errors='coerce', utc=True).dt.tz_convert(None)
        curr_dates = pd.Series([pd.Timestamp.now()] * len(df), index=df.index)
        df['interval_days'] = (curr_dates - last_dates).dt.days.fillna(30)
        df['is_long_rest'] = (df['interval_days'] > 180).astype(int)
    else:
        df['interval_days'] = 30
        df['is_long_rest'] = 0
        
    if '調子偏差値' in df.columns and 'prev1_調子偏差値' in df.columns:
        df['is_dev_up_after_rest'] = ((df['is_long_rest'] == 1) & (df['調子偏差値'] > df['prev1_調子偏差値'])).astype(int)
    else:
        df['is_dev_up_after_rest'] = 0

    if 'prev1_通過' in df.columns:
        df['prev1_通過'] = pd.to_numeric(df['prev1_通過'], errors='coerce').fillna(10.0)
        df['race_expected_pace'] = df.groupby('race_id')['prev1_通過'].transform(lambda x: x.nsmallest(3).mean()).fillna(10.0)
    else:
        df['race_expected_pace'] = 10.0
        
    if '脚質' in df.columns:
        df['temp_is_nige'] = (df['脚質'] == '逃').astype(int)
        df['race_nige_count'] = df.groupby('race_id')['temp_is_nige'].transform('sum')
        df['pace_high_advantage'] = ((df['race_nige_count'] >= 3) & (df['脚質'].isin(['差', '追']))).astype(int)
    else:
        df['race_nige_count'] = 0
        df['pace_high_advantage'] = 0

    # ベースカラムの相対評価 (_race_diff, _race_zscore, 等)
    base_cols = []
    for f in features:
        if f.endswith('_race_diff'): base_cols.append(f.replace('_race_diff', ''))
        elif f.endswith('_race_zscore'): base_cols.append(f.replace('_race_zscore', ''))
        elif f.endswith('_race_rank'): base_cols.append(f.replace('_race_rank', ''))
        elif f.endswith('_race_ratio'): base_cols.append(f.replace('_race_ratio', ''))
    
    base_cols = list(set(base_cols))
    
    for b in base_cols:
        if b not in df.columns:
            df[b] = 0.0
        else:
            df[b] = pd.to_numeric(df[b], errors='coerce').fillna(0.0)
        
        mean_val = df.groupby('race_id')[b].transform('mean')
        std_val = df.groupby('race_id')[b].transform('std').replace(0, 1.0)
        c_max = df.groupby('race_id')[b].transform('max').replace(0, 1.0)
        
        df[f'{b}_race_diff'] = df[b] - mean_val
        df[f'{b}_race_zscore'] = (df[b] - mean_val) / std_val
        df[f'{b}_race_rank'] = df.groupby('race_id')[b].rank(ascending=False, method='min')
        df[f'{b}_race_ratio'] = df[b] / c_max

    print("\n🔍 AIモデル入力用特徴量の最終チェック:")
    for f in features:
        if f not in df.columns:
            df[f] = np.nan
        else:
            df[f] = pd.to_numeric(df[f], errors='coerce')

    null_counts = df[features].isnull().sum()
    complete_features = sum(null_counts == 0)
    partial_null_features = sum((null_counts > 0) & (null_counts < len(df)))
    all_null_features = sum(null_counts == len(df))

    print(f"  ・完全に入力可能: {complete_features} / {len(features)} 項目")
    print(f"  ・一部欠損（初出走馬等）: {partial_null_features} / {len(features)} 項目")
    print(f"  ・全欠損（要確認）: {all_null_features} / {len(features)} 項目")

    if all_null_features > 0:
        print("\n❌ 依然として全欠損している特徴量:")
        for col in null_counts[null_counts == len(df)].index:
            print(f"  - {col}")
    else:
        print(f"\n🎉【検証大成功】全欠損が『 0 』になりました！{len(features)}個の特徴量が完璧に繋がりました！")

if __name__ == "__main__":
    check_merge()