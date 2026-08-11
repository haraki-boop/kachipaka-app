import pandas as pd
import unicodedata
import re
import os
import sys

FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s\u3000]+', '', s)

print("🚀 修正版データ結合処理を開始します...")

if not os.path.exists(FUTURE_CSV) or not os.path.exists(ML_TARGET_CSV):
    print("❌ CSVファイルが見つかりません。")
    sys.exit(1)

try:
    df_future = pd.read_csv(FUTURE_CSV, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_future = pd.read_csv(FUTURE_CSV, encoding='cp932')

try:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')

# 1. 前走由来の混入カラムを future_races から事前に完全削除
dirty_cols = ['place_code', 'course_avg_time', 'course_avg_last3f', 'course_avg_first', 'surface_num', 'today_place_code']
df_future.drop(columns=[c for c in dirty_cols if c in df_future.columns], inplace=True)

# 2. race_id から今日の正確な競馬場コード(place_code)を判定
df_future['place_code'] = df_future['race_id'].astype(str).str[4:6].astype(float)
df_past['place_code'] = df_past['race_id'].astype(str).str[4:6].astype(float)

df_future['surface_num'] = df_future['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2})
df_past['surface_num'] = df_past['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2, '芝→ダート': 1})

# 3. 過去データからコース条件別の正当な平均タイムを計算
course_stats = df_past.groupby(['place_code', 'distance', 'surface_num']).agg({
    'time_seconds': 'mean',
    'last_3f_val': 'mean',
    'first_half_time': 'mean'
}).reset_index()

course_stats.rename(columns={
    'time_seconds': 'true_course_avg_time',
    'last_3f_val': 'true_course_avg_last3f',
    'first_half_time': 'true_course_avg_first'
}, inplace=True)

# 4. 馬の過去能力指数の抽出
df_future['馬名_clean'] = df_future['馬名'].astype(str).apply(clean_horse_name)
df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)

if 'date' in df_past.columns:
    df_past = df_past.sort_values('date')
df_past_latest = df_past.drop_duplicates(subset='馬名_clean', keep='last')

# 過去データから引き継ぐべきでないレース単位のカラムを除外
cols_to_drop = [
    'race_id', 'date', 'race_name', '枠番', '馬番', '馬名', 'sex_code', 'age', 
    '斤量', '騎手', 'オッズ', '人気', '単勝', '着順', 'surface', 'distance', 'condition', 'weather',
    'place_code', 'surface_num', 'time_seconds', 'last_3f_val', 'first_half_time'
] + dirty_cols

cols_to_keep = [c for c in df_past_latest.columns if c not in cols_to_drop]
df_past_latest = df_past_latest[cols_to_keep]

# 既存の指数データカラムとの重複を防ぐ
dup_cols = [c for c in df_past_latest.columns if c in df_future.columns and c != '馬名_clean']
df_future.drop(columns=dup_cols, inplace=True)

# 5. 馬の能力データと正しいコース平均タイムをマージ
df_merged = pd.merge(df_future, df_past_latest, on='馬名_clean', how='left')
df_merged = pd.merge(df_merged, course_stats, on=['place_code', 'distance', 'surface_num'], how='left')

df_merged['course_avg_time'] = df_merged['true_course_avg_time']
df_merged['course_avg_last3f'] = df_merged['true_course_avg_last3f']
df_merged['course_avg_first'] = df_merged['true_course_avg_first']

# 後処理
df_merged.drop(columns=['馬名_clean', 'surface_num', 'true_course_avg_time', 'true_course_avg_last3f', 'true_course_avg_first'], inplace=True, errors='ignore')

df_merged.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
print(f"🎉 結合・補正完了: {FUTURE_CSV} を書き換えました。")