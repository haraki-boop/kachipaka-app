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

print("🚀 3年分のデータを用いた【完全版】結合解析を開始します...")

# 1. 徹底的なファイル存在チェック
if not os.path.exists(FUTURE_CSV):
    print(f"❌ エラー: '{FUTURE_CSV}' が見つかりません。")
    sys.exit(1)
if not os.path.exists(ML_TARGET_CSV):
    print(f"❌ エラー: '{ML_TARGET_CSV}' が見つかりません。")
    sys.exit(1)

# 2. 確実なデータ読み込み（エラーを握りつぶさない）
print(f"📖 {FUTURE_CSV} を読み込んでいます...")
try:
    df_future = pd.read_csv(FUTURE_CSV, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_future = pd.read_csv(FUTURE_CSV, encoding='cp932')

print(f"📖 {ML_TARGET_CSV} を読み込んでいます...")
try:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='utf-8-sig')
except UnicodeDecodeError:
    df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, encoding='cp932')

if df_past.empty:
    print("❌ エラー: ml_target_data.csv の中身が空です。")
    sys.exit(1)

# 3. プレースコードと馬場の統一
PLACE_MAP_REV = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
df_future['place_code'] = df_future['race_id'].astype(str).str[4:6].astype(float)
df_past['place_code'] = df_past['race_id'].astype(str).str[4:6].astype(float)
df_past['surface_num'] = df_past['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2, '芝→ダート': 1})
df_future['surface_num'] = df_future['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2})

# 4. コース条件ごとの正確な平均タイムを算出
print("📊 各コースの基準タイムを算出中...")
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

# 5. 馬の過去能力データの抽出（前走条件の汚染を徹底排除）
print("🐎 馬の能力データを照合中...")
df_future['馬名_clean'] = df_future['馬名'].astype(str).apply(clean_horse_name)
df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)

if 'date' in df_past.columns:
    df_past = df_past.sort_values('date')
df_past_latest = df_past.drop_duplicates(subset='馬名_clean', keep='last')

# future_races に存在する汚染されたカラム名
contaminated_cols = ['course_avg_time', 'course_avg_last3f', 'course_avg_first']
for col in contaminated_cols:
    if col in df_future.columns:
        df_future.drop(columns=[col], inplace=True)

# 過去データから引き継いではいけない項目（今回のレース条件）
cols_to_drop = [
    'race_id', 'date', 'race_name', '枠番', '馬番', '馬名', 'sex_code', 'age', 
    '斤量', '騎手', 'オッズ', '人気', '単勝', '着順', 'surface', 'distance', 'condition', 'weather',
    'place_code', 'surface_num'
]
cols_to_keep = [c for c in df_past_latest.columns if c not in cols_to_drop + contaminated_cols]
df_past_latest = df_past_latest[cols_to_keep]

# 6. 未来データに馬の能力を結合
print("🔗 データを結合しています...")
# 既存の指数データ（前走引き継ぎ分）を一旦削除して、完全にクリーンな状態から結合
duplicate_cols = [col for col in df_past_latest.columns if col in df_future.columns and col != '馬名_clean']
df_future.drop(columns=duplicate_cols, inplace=True, errors='ignore')

df_merged = pd.merge(df_future, df_past_latest, on='馬名_clean', how='left')

# 7. 「今日のレース条件」に合致する正しい平均タイムを結合
df_merged = pd.merge(df_merged, course_stats, on=['place_code', 'distance', 'surface_num'], how='left')

df_merged['course_avg_time'] = df_merged['true_course_avg_time']
df_merged['course_avg_last3f'] = df_merged['true_course_avg_last3f']
df_merged['course_avg_first'] = df_merged['true_course_avg_first']

# お掃除
cols_to_clean = ['馬名_clean', 'surface_num', 'true_course_avg_time', 'true_course_avg_last3f', 'true_course_avg_first']
df_merged.drop(columns=[c for c in cols_to_clean if c in df_merged.columns], inplace=True)

# 8. 保存して最終確認
df_merged.to_csv(FUTURE_CSV, index=False, encoding='utf-8-sig')

# ちゃんと直ったか新潟8Rをチェックして画面に出す
print("\n=== 修復確認: 新潟8R (驀進特別) のコース基準タイム ===")
check_df = df_merged[(df_merged['race_id'].astype(str).str.contains('04')) & (df_merged['race_id'].astype(str).str.endswith('08'))]
if not check_df.empty:
    print(check_df[['馬名', 'distance', 'course_avg_time']].head(5))
    print(f"\n🎉 完璧なデータ復元が完了しました！ {FUTURE_CSV} を上書きしました。")
else:
    print("⚠️ 新潟8Rが見つかりませんでしたが、処理は完了しました。")