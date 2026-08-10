import pandas as pd
import unicodedata
import re

FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s\u3000]+', '', s)

print("🚀 3年分のデータを用いた【完全版】結合解析を開始します...")

# 1. データの読み込み
try:
    df_future = pd.read_csv(FUTURE_CSV, encoding='utf-8-sig')
except Exception:
    df_future = pd.read_csv(FUTURE_CSV, encoding='cp932')

try:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
except Exception:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')

# 2. 【最重要】3年分のデータから、コースごとの「正しい平均タイム」を再計算する
print("📊 各コースの基準タイムを3年分のデータから算出中...")
course_stats = df_past.groupby(['place_code', 'distance', 'surface']).agg({
    'time_seconds': 'mean',
    'last_3f_val': 'mean',
    'first_half_time': 'mean'
}).reset_index()
course_stats.rename(columns={
    'time_seconds': 'true_course_avg_time',
    'last_3f_val': 'true_course_avg_last3f',
    'first_half_time': 'true_course_avg_first'
}, inplace=True)

# 3. 馬名のクレンジング
df_future['馬名_clean'] = df_future['馬名'].apply(clean_horse_name)
df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)

# 4. 馬の「過去の個人能力（指数）」だけを抽出
if 'date' in df_past.columns:
    df_past = df_past.sort_values('date')
df_past_latest = df_past.drop_duplicates(subset='馬名_clean', keep='last')

# ❌ 結合してはいけない「前走のレース条件・前走のコース平均」を除外
cols_to_drop = [
    'race_id', 'date', 'race_name', '枠番', '馬番', '馬名', 'sex_code', 'age', 
    '斤量', '騎手', 'オッズ', '人気', '単勝', '着順', 'surface', 'distance', 'condition', 'weather',
    'place_code', 'course_avg_time', 'course_avg_last3f', 'course_avg_first' # これらが真犯人
]
cols_to_keep = [c for c in df_past_latest.columns if c not in cols_to_drop]
df_past_latest = df_past_latest[cols_to_keep]

# 5. まず「馬の個人能力」を未来データに結合
df_merged = pd.merge(df_future, df_past_latest, on='馬名_clean', how='left')
df_merged.drop(columns=['馬名_clean'], inplace=True)

# 6. 次に「今日のレース条件に合った、正しいコース平均タイム」を結合
# （※future_races側にplace_codeが無い場合を考慮して再作成）
PLACE_MAP_REV = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
if 'place_code' not in df_merged.columns:
    df_merged['place_code'] = df_merged['race_id'].astype(str).str[4:6].astype(float)
    
df_merged['surface'] = df_merged['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2})

# コース基準値をマージ
df_merged = pd.merge(df_merged, course_stats, on=['place_code', 'distance', 'surface'], how='left')

# AIが読み取るカラム名に上書き
df_merged['course_avg_time'] = df_merged['true_course_avg_time']
df_merged['course_avg_last3f'] = df_merged['true_course_avg_last3f']
df_merged['course_avg_first'] = df_merged['true_course_avg_first']

# 不要になった中間カラムを削除
df_merged.drop(columns=['true_course_avg_time', 'true_course_avg_last3f', 'true_course_avg_first'], inplace=True)
# surfaceを文字に戻す（アプリ側での表示用）
df_merged['surface'] = df_merged['surface'].replace({0.0: '芝', 1.0: 'ダート', 2.0: '障害'})

# 7. 保存
df_merged.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
print(f"🎉 完璧なデータ復元が完了しました！ {FUTURE_CSV} を上書きしました。")