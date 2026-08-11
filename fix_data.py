import pandas as pd
import numpy as np
import os
import sys

FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"

if not os.path.exists(FUTURE_CSV):
    print(f"❌ エラー: {FUTURE_CSV} が見つかりません。")
    sys.exit(1)

print(f"📖 {FUTURE_CSV} の読み込みとクレンジングを開始します...")

try:
    df = pd.read_csv(FUTURE_CSV, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(FUTURE_CSV, encoding='cp932')

# 1. 過去のマージで重複・肥大化した '_x', '_y' カラムの整理
clean_cols = [c for c in df.columns if not (c.endswith('_x') or c.endswith('_y'))]
df = df[clean_cols].copy()

# 2. 競馬場コードと馬場コードの判定
df['place_code'] = df['race_id'].astype(str).str[4:6].astype(float)
df['surface_num'] = df['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2})

# 3. 2233%などの暴走した勝率・成績データの再計算および正常化 (0.0 〜 1.0 の範囲に抑える)
for prefix in ['horse', 'jockey']:
    runs_col = f'{prefix}_runs'
    wins_col = f'{prefix}_wins'
    rate_col = f'{prefix}_win_rate'
    
    if runs_col in df.columns and wins_col in df.columns:
        mask = df[runs_col] > 0
        df.loc[mask, rate_col] = df.loc[mask, wins_col] / df.loc[mask, runs_col]
        df.loc[~mask, rate_col] = 0.0
        df[rate_col] = df[rate_col].clip(0.0, 1.0)

if 'jp_runs' in df.columns and 'jp_wins' in df.columns:
    mask = df['jp_runs'] > 0
    df.loc[mask, 'jockey_track_win_rate'] = df.loc[mask, 'jp_wins'] / df.loc[mask, 'jp_runs']
    df.loc[~mask, 'jockey_track_win_rate'] = 0.0
    df['jockey_track_win_rate'] = df['jockey_track_win_rate'].clip(0.0, 1.0)

# 4. 過去データ(ml_target_data.csv) がある場合はそこから正当なコース平均値を補填
if os.path.exists(ML_TARGET_CSV):
    try:
        df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')
        
    df_past['place_code'] = df_past['race_id'].astype(str).str[4:6].astype(float)
    df_past['surface_num'] = df_past['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2, '芝→ダート': 1})
    
    stats = df_past.groupby(['place_code', 'distance', 'surface_num']).agg({
        'time_seconds': 'mean',
        'last_3f_val': 'mean',
        'first_half_time': 'mean'
    }).reset_index()
    
    stats.rename(columns={
        'time_seconds': 'calc_course_avg_time',
        'last_3f_val': 'calc_course_avg_last3f',
        'first_half_time': 'calc_course_avg_first'
    }, inplace=True)
    
    df = pd.merge(df, stats, on=['place_code', 'distance', 'surface_num'], how='left')
    
    for c, calc_c in [('course_avg_time', 'calc_course_avg_time'), 
                      ('course_avg_last3f', 'calc_course_avg_last3f'), 
                      ('course_avg_first', 'calc_course_avg_first')]:
        if c in df.columns:
            df[c] = df[c].fillna(df[calc_c])
        else:
            df[c] = df[calc_c]
        df.drop(columns=[calc_c], inplace=True, errors='ignore')

# 5. ダート等でコース平均値が万が一残欠損(NaN)の場合、同一距離・馬場の基準平均値で安全に補填
dist_surf_means = df.groupby(['distance', 'surface_num'])['course_avg_time'].mean().to_dict()

def fill_remaining_avg(row):
    val = row.get('course_avg_time')
    if pd.notna(val) and val > 0:
        return val
    key = (row['distance'], row['surface_num'])
    if key in dist_surf_means and pd.notna(dist_surf_means[key]):
        return dist_surf_means[key]
    return row['distance'] * (0.063 if row['surface_num'] == 1 else 0.059)

df['course_avg_time'] = df.apply(fill_remaining_avg, axis=1)

# 不用な中間カラムのクレンジング
df.drop(columns=['place_code', 'surface_num'], inplace=True, errors='ignore')

# 6. 保存
df.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')

print(f"✅ 修正完了: {FUTURE_CSV} の異常値・欠損・重複カラムをすべて正常化しました。")

# 7. 中京5Rの修正確認
c5 = df[(df['race_id'].astype(str).str.contains('07')) & (df['race_id'].astype(str).str.endswith('05'))]
if not c5.empty:
    print("\n=== 修正確認: 中京5R (ダート1400m) ===")
    print(c5[['馬番', '馬名', 'jockey_win_rate', 'course_avg_time']].head(5).to_string())