import os
import re
import joblib
import unicodedata
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

MODEL_FILE = "keiba_ai_model.pkl"
FUTURE_CSV = "future_races.csv"
CACHE_FILE = "app_cache.pkl"

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

        lgb_pred = self.lgb_model.predict(X_num)
        lgb_pred = (lgb_pred - np.mean(lgb_pred)) / (np.std(lgb_pred) + 1e-8)

        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_num))
        xgb_pred = (xgb_pred - np.mean(xgb_pred)) / (np.std(xgb_pred) + 1e-8)

        cat_pred = self.cat_model.predict(X_num)
        cat_pred = (cat_pred - np.mean(cat_pred)) / (np.std(cat_pred) + 1e-8)

        w1, w2, w3 = self.weights
        return w1 * lgb_pred + w2 * xgb_pred + w3 * cat_pred

import __main__
__main__.EnsembleModel = EnsembleModel

def clean_name(text):
    if pd.isna(text): return ""
    s = unicodedata.normalize('NFKC', str(text))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def main():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(FUTURE_CSV) or not os.path.exists(CACHE_FILE):
        print(f"❌ 必要なファイルが不足しています。（{MODEL_FILE}, {FUTURE_CSV}, {CACHE_FILE} が必要）")
        return

    print("🔄 モデル、キャッシュ辞書、最新出馬表を読み込んでいます...")
    model_data = joblib.load(MODEL_FILE)
    model = model_data['model']
    features = model_data['features']
    
    cache = joblib.load(CACHE_FILE)
    df_raw = pd.read_csv(FUTURE_CSV, low_memory=False)
    
    if 'race_id' not in df_raw.columns or df_raw.empty:
        print("❌ 対象となる未来のレースデータが存在しません。")
        return

    print("⚡ 出馬表に過去の成績・指数キャッシュを高速結合中 (フルスペック201特徴量)...")
    df = df_raw.copy()
    
    # 1. 基本データ正規化
    df['race_id'] = df['race_id'].astype(str).str.zfill(12)
    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_name)
    df['騎手_clean'] = df['騎手'].astype(str).apply(clean_name)
    df['調教師_clean'] = df['調教師'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).apply(clean_name)
    
    places = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京',
              '06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    df['place_name'] = df['race_id'].astype(str).str[4:6].map(places).fillna('不明')
    
    if 'distance' in df.columns:
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(1600.0)
    else:
        df['distance'] = 1600.0
        
    df['is_right_turn'] = df['place_name'].isin(['中山', '阪神', '京都', '小倉', '福島', '札幌', '函館']).astype(int)
    df['is_steep_hill'] = df['place_name'].isin(['中山', '阪神']).astype(int)

    # 2. キャッシュ全辞書の完全結合
    for key, value in cache.items():
        if key == 'predictions': continue
        if isinstance(value, dict) and len(value) > 0:
            first_val = next(iter(value.values()))
            if isinstance(first_val, dict):
                temp_df = pd.DataFrame.from_dict(value, orient='index').reset_index()
                temp_df = temp_df.rename(columns={'index': '馬名_clean'})
                cols_to_use = temp_df.columns.difference(df.columns).tolist() + ['馬名_clean']
                df = pd.merge(df, temp_df[cols_to_use], on='馬名_clean', how='left')

    # 3. 騎手・調教師エンコーディング
    df['騎手_win_rate'] = df['騎手_clean'].map(cache.get('jockey_win_map', {})).fillna(0.05)
    df['騎手_place_rate'] = df['騎手_clean'].map(cache.get('jockey_place_map', {})).fillna(0.15)
    df['trainer_win_rate'] = df['調教師_clean'].map(cache.get('trainer_win_map', {})).fillna(0.05)
    df['trainer_place_rate'] = df['調教師_clean'].map(cache.get('trainer_place_map', {})).fillna(0.15)

    df['jp_key'] = df['騎手_clean'] + "_" + df['place_name']
    jp_win_flat = {f"{k[0]}_{k[1]}": v for k, v in cache.get('jp_win_map', {}).items()}
    jp_place_flat = {f"{k[0]}_{k[1]}": v for k, v in cache.get('jp_place_map', {}).items()}
    df['jockey_place_win_rate'] = df['jp_key'].map(jp_win_flat).fillna(0.05)
    df['jockey_place_place_rate'] = df['jp_key'].map(jp_place_flat).fillna(0.15)

    df['tp_key'] = df['調教師_clean'] + "_" + df['place_name']
    tp_win_flat = {f"{k[0]}_{k[1]}": v for k, v in cache.get('tp_win_map', {}).items()}
    df['trainer_place_win_rate'] = df['tp_key'].map(tp_win_flat).fillna(0.05)

    # 4. 動的特徴量
    if 'last_date' in df.columns:
        last_dates = pd.to_datetime(df['last_date'], errors='coerce', utc=True).dt.tz_convert(None)
        curr_dates = pd.Series([pd.Timestamp.now()] * len(df), index=df.index)
        df['interval_days'] = (curr_dates - last_dates).dt.days.fillna(30)
    else:
        df['interval_days'] = 30

    if 'prev_1c' in df.columns:
        df['prev_1c'] = pd.to_numeric(df['prev_1c'], errors='coerce').fillna(10.0)
        df['race_expected_pace'] = df.groupby('race_id')['prev_1c'].transform('mean').fillna(10.0)
        df['pace_advantage'] = df['prev_1c'] - df['race_expected_pace']
    else:
        df['race_expected_pace'] = 10.0
        df['pace_advantage'] = 0.0

    if '斤量' in df.columns:
        df['kinryo_num'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(55.0)
    else:
        df['kinryo_num'] = 55.0

    for idx_type in ['pace_idx', 'last3f_idx', 'start_idx']:
        cols = [f'prev{i}_{idx_type}' for i in range(1, 6)]
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            df[f'max_{idx_type}'] = df[valid_cols].astype(float).max(axis=1).fillna(0.0)
            if f'prev1_{idx_type}' in df.columns:
                df[f'ratio_to_max_{idx_type}'] = pd.to_numeric(df[f'prev1_{idx_type}'], errors='coerce') / df[f'max_{idx_type}'].replace(0, 1.0)
                df[f'ratio_to_max_{idx_type}'] = df[f'ratio_to_max_{idx_type}'].fillna(0.0)
        else:
            df[f'max_{idx_type}'] = 0.0
            df[f'ratio_to_max_{idx_type}'] = 0.0

    base_cols = []
    for f in features:
        if f.endswith('_race_diff'): base_cols.append(f.replace('_race_diff', ''))
        elif f.endswith('_race_zscore'): base_cols.append(f.replace('_race_zscore', ''))
        elif f.endswith('_race_rank'): base_cols.append(f.replace('_race_rank', ''))
        elif f.endswith('_race_ratio'): base_cols.append(f.replace('_race_ratio', ''))
    
    base_cols = list(set(base_cols))
    for b in base_cols:
        if b not in df.columns: df[b] = 0.0
        else: df[b] = pd.to_numeric(df[b], errors='coerce').fillna(0.0)
        mean_val = df.groupby('race_id')[b].transform('mean')
        std_val = df.groupby('race_id')[b].transform('std').replace(0, 1.0)
        df[f'{b}_race_diff'] = df[b] - mean_val
        df[f'{b}_race_zscore'] = (df[b] - mean_val) / std_val
        df[f'{b}_race_rank'] = df.groupby('race_id')[b].rank(ascending=False, method='min')
        df[f'{b}_race_ratio'] = df[b] / mean_val.replace(0, 1.0)

    # 🔍 特徴量接続ログ
    print("\n🔍 AIモデル入力用特徴量の接続状態チェック:")
    for f in features:
        if f not in df.columns: df[f] = np.nan
        else: df[f] = pd.to_numeric(df[f], errors='coerce')

    null_counts = df[features].isnull().sum()
    complete_features = sum(null_counts == 0)
    partial_null_features = sum((null_counts > 0) & (null_counts < len(df)))
    all_null_features = sum(null_counts == len(df))

    print(f"  ・完全に入力可能: {complete_features} / {len(features)} 項目")
    print(f"  ・一部欠損（初出走馬等）: {partial_null_features} / {len(features)} 項目")
    print(f"  ・全欠損: {all_null_features} / {len(features)} 項目")

    if all_null_features > 0:
        print("❌ 全欠損している特徴量が存在します。確認してください。")
    else:
        print("🎉【特徴量結合完了】全欠損ゼロで完璧にデータが接続されました！")

    # 5. モデル推論
    X_future = pd.DataFrame(index=df.index)
    for f in features:
        X_future[f] = pd.to_numeric(df[f], errors='coerce').fillna(0.0)

    print("\n🤖 勝ちパカくんの最強頭脳が未来のレースを推論中...")
    df['raw_score'] = model.predict(X_future)

    # 6. 一括計算 & 構造化（安全な1回ループ設計）
    pred_dict = {}
    history_records = []

    print("\n" + "="*85)
    print("🏇 【勝ちパカくん】本気配察知・適応買い目 予想レポート (🔥強気モード)")
    print("="*85)

    for race_id, group in df.groupby('race_id'):
        group = group.copy()
        raw_scores = group['raw_score'].values
        s_std = np.std(raw_scores)
        if pd.notna(s_std) and s_std > 0:
            z_scores = (raw_scores - np.mean(raw_scores)) / s_std
            group['win_prob'] = (1.0 / (1.0 + np.exp(-1.2 * z_scores))) * 0.35 + 0.01
        else:
            group['win_prob'] = 0.10
        
        min_p, max_p = group['win_prob'].min(), group['win_prob'].max()
        if max_p > min_p:
            group['ai_score'] = (50 + (group['win_prob'] - min_p) / (max_p - min_p) * 100).round().astype(int)
        else:
            group['ai_score'] = 100

        # 脚質判定
        temp_val = None
        ascending = True
        if 'prev_1c' in group.columns and pd.to_numeric(group['prev_1c'], errors='coerce').nunique() > 1:
            temp_val = pd.to_numeric(group['prev_1c'], errors='coerce')
            ascending = True
        elif 'ema5_start_idx' in group.columns and pd.to_numeric(group['ema5_start_idx'], errors='coerce').nunique() > 1:
            temp_val = pd.to_numeric(group['ema5_start_idx'], errors='coerce')
            ascending = False

        if temp_val is not None:
            rank = temp_val.rank(ascending=ascending, method='min')
        else:
            rank = pd.to_numeric(group.get('馬番', group.index), errors='coerce').rank(ascending=True, method='min')

        valid_len = len(group)
        styles = []
        for r in rank:
            pct = r / valid_len if valid_len > 0 else 0
            if pct <= 0.25: styles.append("逃げ")
            elif pct <= 0.50: styles.append("先行")
            elif pct <= 0.75: styles.append("差し")
            else: styles.append("追込")
        group['脚質'] = styles

        # ソートして印を割り当て
        group = group.sort_values(by=['win_prob'], ascending=False).reset_index(drop=True)
        marks = ["◎", "◯", "▲", "△", "☆1", "☆2"]
        group['印'] = "消"
        for i in range(min(len(group), len(marks))):
            group.loc[i, '印'] = marks[i]

        probs = group['win_prob'].values
        p1 = probs[0] if len(probs)>0 else 0
        p2 = probs[1] if len(probs)>1 else 0
        p3 = probs[2] if len(probs)>2 else 0
        p4 = probs[3] if len(probs)>3 else 0.05
        gap_1_2, gap_1_3 = p1 - p2, p1 - p3

        if gap_1_2 >= 0.07:
            pat = "① 1強気配 (軸圧倒) 🔥勝負!"
            rec_ticket = "3連単 1着固定 (6点)"
            buy_detail = f"1着: {group.loc[0, '馬番']}(◎) -> 2・3着: {group.loc[1, '馬番']}(◯), {group.loc[2, '馬番']}(▲), {group.loc[3, '馬番']}(△)"
        elif gap_1_2 < 0.035 and gap_1_3 >= 0.06:
            pat = "② 2強気配 (頭分け対抗) 🔥勝負!"
            rec_ticket = "3連単 ダブル軸 (12点)"
            buy_detail = f"1着: {group.loc[0, '馬番']}(◎), {group.loc[1, '馬番']}(◯) -> 2着: ◎, ◯, {group.loc[2, '馬番']}(▲) -> 3着: ◎, ◯, ▲, {group.loc[3, '馬番']}(△)"
        elif (p1 - p4) < 0.08:
            pat = "④ 波乱気配 (大混戦)"
            rec_ticket = "3連複 5頭BOX (10点)"
            buy_detail = "BOX: " + ", ".join([f"{group.loc[k, '馬番']}({group.loc[k, '印']})" for k in range(min(5, len(group)))])
        else:
            pat = "③ 混戦気配 (標準展開)"
            rec_ticket = "3連複 ◎1頭軸流し (10点)"
            buy_detail = f"軸: {group.loc[0, '馬番']}(◎) -> 相手: " + ", ".join([f"{group.loc[k, '馬番']}({group.loc[k, '印']})" for k in range(1, min(6, len(group)))])

        # PKL用の辞書構造を作成
        horses_info = {}
        for _, row in group.iterrows():
            horses_info[row['馬名_clean']] = {
                '馬番': row.get('馬番'),
                'win_prob': row.get('win_prob'),
                'ai_score': row.get('ai_score'),
                '脚質': row.get('脚質'),
                '印': row.get('印')
            }

        pred_dict[str(race_id)] = {
            'pattern': pat,
            'ticket': rec_ticket,
            'buy_detail': buy_detail,
            'horses': horses_info
        }

        # ターミナル表示
        r_name = group['race_name'].iloc[0] if 'race_name' in group.columns and pd.notna(group['race_name'].iloc[0]) else f"Race {race_id}"
        date_str = group['date'].iloc[0] if 'date' in group.columns and pd.notna(group['date'].iloc[0]) else ""

        print(f"\n📍 レースID: {race_id} | {date_str} {r_name}")
        print(f"  ├ 🧠 勝負気配  : {pat}")
        print(f"  ├ 🎟️ 推奨買い目: {rec_ticket}")
        print(f"  ├ 📝 買目詳細  : {buy_detail}")
        top_str = " | ".join([f"{group.iloc[k]['印']}:{group.iloc[k]['馬名']}({group.iloc[k]['馬番']})" for k in range(min(4, len(group)))])
        print(f"  └ 🐴 上位評価  : {top_str}")

        history_records.append({
            'race_id': race_id, 'date': date_str, 'race_name': r_name,
            'pattern': pat, 'ticket': rec_ticket, 'buy_detail': buy_detail
        })

    # 7. app_cache.pkl の predictions キーを安全に上書き
    cache['predictions'] = pred_dict
    joblib.dump(cache, CACHE_FILE)
    print(f"\n✅ 予想結果・脚質・買い目を '{CACHE_FILE}' (app_cache.pkl) 内に無事保存完了！")

    if history_records:
        pd.DataFrame(history_records).to_csv("prediction_history.csv", index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()