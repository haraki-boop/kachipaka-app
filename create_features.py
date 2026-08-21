import pandas as pd
import numpy as np
import os
import joblib

INPUT_CSV = "ml_target_data.csv"
OUTPUT_CSV = "ml_target_data.csv"

def calc_custom_index(val, m, s):
    if pd.isna(val) or s == 0: return 50.0
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
    # カラムが存在しない場合のエラーを回避
    df['着順'] = pd.to_numeric(df.get('着順'), errors='coerce')
    df = df.dropna(subset=['着順'])
    df['is_win'] = (df['着順'] == 1).astype(int)

    # タイムと上がりの列名を柔軟に取得
    time_col = df.get('タイム', df.get('time', pd.Series(np.nan, index=df.index)))
    last3f_col = df.get('上がり3F', df.get('上がり', df.get('last3f', pd.Series(np.nan, index=df.index))))

    df['タイム'] = pd.to_numeric(time_col, errors='coerce')
    df['上がり3F'] = pd.to_numeric(last3f_col, errors='coerce')

    print("Calculating Race Stats...")
    if 'race_id' in df.columns:
        race_stats = df.groupby('race_id').agg(
            race_avg_time=('タイム', 'mean'),
            race_std_time=('タイム', 'std'),
            race_avg_last3f=('上がり3F', 'mean'),
            race_std_last3f=('上がり3F', 'std')
        ).reset_index()
        df = pd.merge(df, race_stats, on='race_id', how='left')
    else:
        df['race_avg_time'], df['race_std_time'], df['race_avg_last3f'], df['race_std_last3f'] = np.nan, np.nan, np.nan, np.nan

    print("Calculating Custom Indices...")
    df['my_time_idx'] = df.apply(lambda r: calc_custom_index(r['タイム'], r['race_avg_time'], r['race_std_time']), axis=1)
    df['my_last3f_idx'] = df.apply(lambda r: calc_custom_index(r['上がり3F'], r['race_avg_last3f'], r['race_std_last3f']), axis=1)
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