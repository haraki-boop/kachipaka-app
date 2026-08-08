import pandas as pd
import numpy as np
import os
import re
import time

INPUT_CSV = "enhanced_keiba_data.csv" if os.path.exists("enhanced_keiba_data.csv") else "cleaned_keiba_data.csv"
OUTPUT_CSV = "ml_target_data.csv" 

def time_to_sec(t):
    if pd.isna(t): return np.nan
    parts = str(t).strip().split(':')
    if len(parts) == 2: return float(parts[0]) * 60 + float(parts[1])
    try: return float(parts[0])
    except: return np.nan

def get_first_position(p):
    """ 通過順位（例：03-04-05）から最初のポジション（3）を取得してスタート力を測る """
    if pd.isna(p): return np.nan
    nums = re.findall(r'\d+', str(p))
    return float(nums[0]) if nums else np.nan

def main():
    print(f"📥 {INPUT_CSV} を読み込んでいます...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='cp932')

    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    df['is_win'] = (df['着順'] == 1).astype(int)
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['place_code'] = df['race_id'].astype(str).str[4:6]
    
    # タイムと上がりの数値化
    if 'タイム' in df.columns:
        df['time_seconds'] = df['タイム'].apply(time_to_sec)
    if '上り' in df.columns:
        df['last_3f_val'] = pd.to_numeric(df['上り'], errors='coerce')
        
    # 前半・追走タイムの算出 (走破タイム - 上がり3Fタイム)
    df['first_half_time'] = df['time_seconds'] - df['last_3f_val']

    print("🧮 欠損している各指数を自前で近似計算しています...")
    # コース別（競馬場 × 芝/ダート × 距離）の平均タイムを出す
    if 'surface' in df.columns and 'distance' in df.columns:
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        course_grp = df.groupby(['place_code', 'surface', 'distance'])
        df['course_avg_time'] = course_grp['time_seconds'].transform('mean')
        df['course_avg_last3f'] = course_grp['last_3f_val'].transform('mean')
        df['course_avg_first'] = course_grp['first_half_time'].transform('mean')

        # 【AIオリジナル指数】 平均より1秒速ければ+10ポイント、基準は80
        df['my_time_idx'] = (df['course_avg_time'] - df['time_seconds']) * 10 + 80
        df['my_last3f_idx'] = (df['course_avg_last3f'] - df['last_3f_val']) * 10 + 80
        df['my_pace_idx'] = (df['course_avg_first'] - df['first_half_time']) * 10 + 80
    else:
        df['my_time_idx'], df['my_last3f_idx'], df['my_pace_idx'] = np.nan, np.nan, np.nan

    # スタート指数の計算 (最初の通過順位から計算。1番手なら高く、後ろなら低く)
    if '通過' in df.columns:
        df['first_pos'] = df['通過'].apply(get_first_position)
        df['my_start_idx'] = 100 - (df['first_pos'] * 2)
    else:
        df['my_start_idx'] = np.nan

    # 時系列に並び替え (カンニング防止の絶対条件)
    df = df.sort_values(by='date_parsed').reset_index(drop=True)

    print("🐴 馬・騎手の過去実績（自作指数を含む）を計算中...")
    horse_groups = df.groupby('馬名')
    df['horse_runs'] = horse_groups.cumcount()
    df['horse_wins'] = horse_groups['is_win'].cumsum().shift(1).fillna(0)
    df['horse_win_rate'] = np.where(df['horse_runs'] > 0, df['horse_wins'] / df['horse_runs'], 0.0)
    df['prev_rank'] = horse_groups['着順'].shift(1)
    
    # 指数も「そのレースより前の過去の平均」を計算してカンニングを防ぐ
    df['horse_avg_time_idx'] = horse_groups['my_time_idx'].transform(lambda x: x.shift(1).expanding().mean())
    df['horse_avg_last3f_idx'] = horse_groups['my_last3f_idx'].transform(lambda x: x.shift(1).expanding().mean())
    df['horse_avg_pace_idx'] = horse_groups['my_pace_idx'].transform(lambda x: x.shift(1).expanding().mean())
    df['horse_avg_start_idx'] = horse_groups['my_start_idx'].transform(lambda x: x.shift(1).expanding().mean())

    jockey_groups = df.groupby('騎手')
    df['jockey_runs'] = jockey_groups.cumcount()
    df['jockey_wins'] = jockey_groups['is_win'].cumsum().shift(1).fillna(0)
    df['jockey_win_rate'] = np.where(df['jockey_runs'] > 0, df['jockey_wins'] / df['jockey_runs'], 0.0)

    jockey_place_groups = df.groupby(['騎手', 'place_code'])
    df['jp_runs'] = jockey_place_groups.cumcount()
    df['jp_wins'] = jockey_place_groups['is_win'].cumsum().shift(1).fillna(0)
    df['jockey_track_win_rate'] = np.where(df['jp_runs'] > 0, df['jp_wins'] / df['jp_runs'], 0.0)

    print(f"💾 計算完了！最強の特徴量データを {OUTPUT_CSV} に保存しています...")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("✅ 完了しました！")

if __name__ == "__main__":
    main()