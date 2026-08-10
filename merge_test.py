import pandas as pd
import unicodedata
import re

FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s\u3000]+', '', s)

print("🚀 結合テストを開始します...")

# 1. 未来の出馬表（8/9のデータ）を読み込み
try:
    df_future = pd.read_csv(FUTURE_CSV, encoding='utf-8-sig')
except Exception:
    df_future = pd.read_csv(FUTURE_CSV, encoding='cp932')
print(f"📖 {FUTURE_CSV} を読み込みました。")

# 2. 過去データ（3年分）を「読み取り専用」で読み込み（絶対に上書きしません）
try:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
except Exception:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')
print(f"📖 {ML_TARGET_CSV} を安全に読み込みました。")

# 3. 馬名を綺麗にして照合の準備
df_future['馬名_clean'] = df_future['馬名'].apply(clean_horse_name)
df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)

# 4. 過去データから各馬の「一番新しいレース」のデータだけを抽出
if 'date' in df_past.columns:
    df_past = df_past.sort_values('date')
df_past_latest = df_past.drop_duplicates(subset='馬名_clean', keep='last')

# 5. 上書きしてはいけない基本情報（未来のレース条件）を除外し、指数データのみを残す
cols_to_drop = [
    'race_id', 'date', 'race_name', '枠番', '馬番', '馬名', 'sex_code', 'age', 
    '斤量', '騎手', 'オッズ', '人気', '単勝', '着順', 'surface', 'distance', 'condition', 'weather'
]
cols_to_keep = [c for c in df_past_latest.columns if c not in cols_to_drop]
df_past_latest = df_past_latest[cols_to_keep]

# 6. 未来の出馬表に過去の指数を合体（左結合）
df_merged = pd.merge(df_future, df_past_latest, on='馬名_clean', how='left')
df_merged.drop(columns=['馬名_clean'], inplace=True)

# 7. 完成したデータを future_races.csv に上書き保存
df_merged.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')
print(f"🎉 結合完了！ {FUTURE_CSV} に指数データを書き込みました。")
print(f"⚠️ {ML_TARGET_CSV} は一切変更されていません。")