import pandas as pd
import numpy as np
import os
import joblib
import re

INPUT_CSV = "ml_target_data.csv"
OUTPUT_CSV = "ml_target_data.csv"

# 🛑 【追加】タイム文字列（例: 1:33.2）を秒数（93.2）に変換する関数
def parse_time_str(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    m = re.match(r'(?:(\d+)[:.])?(\d{1,2})\.(\d+)', s)
    if m:
        mins = int(m.group(1)) if m.group(1) else 0
        secs = int(m.group(2))
        ms = float('0.' + m.group(3))
        return mins * 60 + secs + ms
    try:
        return float(s)
    except:
        return np.nan

def calc_custom_index(val, m, s):
    if pd.isna(val) or pd.isna(s) or s == 0: return 50.0
    return 50.0 + ((m - val) / s) * 10.0

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_CSV, low_memory=False, encoding='cp932')

    print("Cleaning data...")
    df['着順'] = pd.to_numeric(df.get('着順'), errors='coerce')
    df = df.dropna(subset=['着順']).copy()
    df['is_win'] = (df['着順'] == 1).astype(int)

    time_col = df.get('タイム', df.get('time', pd.Series(np.nan, index=df.index)))
    last3f_col = df.get('上がり3F', df.get('上がり', df.get('last3f', pd.Series(np.nan, index=df.index))))

    # 🛑 ここで文字列変換処理を適用
    df['タイム'] = time_col.apply(parse_time_str)
    df['上がり3F'] = last3f_col.apply(parse_time_str)

    drop_cols = ['race_avg_time', 'race_std_time', 'race_avg_last3f', 'race_std_last3f', 'jockey_win_power']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.drop(columns=[c for c in df.columns if c.endswith('_x') or c.endswith('_y')])

    print("Calculating Race Stats...")
    if 'race_id' in df.columns:
        race_stats = df.groupby('race_id').agg(
            race_avg_time=('タイム', 'mean'),
            race_std_time=('タイム', 'std'),
            race_avg_last3f=('上がり3F', 'mean'),
            race_std_last3f=('上がり3F', 'std')
        ).reset_index()
        df = pd.merge(df, race_stats, on='race_id', how='left')
    
    safe_cols = ['race_avg_time', 'race_std_time', 'race_avg_last3f', 'race_std_last3f']
    for c in safe_cols:
        if c not in df.columns:
            df[c] = np.nan

    print("Calculating Custom Indices...")
    df['my_time_idx'] = df.apply(lambda r: calc_custom_index(r.get('タイム'), r.get('race_avg_time'), r.get('race_std_time')), axis=1)
    df['my_last3f_idx'] = df.apply(lambda r: calc_custom_index(r.get('上がり3F'), r.get('race_avg_last3f'), r.get('race_std_last3f')), axis=1)
    df['my_pace_idx'] = df['my_time_idx'] * 0.4 + df['my_last3f_idx'] * 0.6
    df['my_start_idx'] = df['my_time_idx'] * 0.7 + df['my_last3f_idx'] * 0.3

    print("Calculating Jockey Stats...")
    if '騎手' in df.columns:
        jockey_stats = df.groupby('騎手')['is_win'].mean().reset_index()
        jockey_stats.rename(columns={'is_win': 'jockey_win_power'}, inplace=True)
        df = pd.merge(df, jockey_stats, on='騎手', how='left')

    print("Encoding categories...")
    if 'surface' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le_surf = LabelEncoder()
        df['surface_code'] = le_surf.fit_transform(df['surface'].astype(str))
        joblib.dump(le_surf, "le_surf.pkl")
    if 'condition' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le_cond = LabelEncoder()
        df['condition_code'] = le_cond.fit_transform(df['condition'].astype(str))
        joblib.dump(le_cond, "le_cond.pkl")
    if 'sex_code' in df.columns:
        df['sex_code'] = df['sex_code'].map({'牡': 0, '牝': 1, 'セ': 2}).fillna(0)

    print(f"Saving output to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print("Done!")

if __name__ == "__main__":
    main()