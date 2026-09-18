import os
import re
import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = 'keiba_database.csv'
OUTPUT_FILE = 'ml_target_data.csv'

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type) and not pd.api.types.is_bool_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if pd.isna(c_min) or pd.isna(c_max): continue
            
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

def clean_data(df):
    print("  -> [1/5] クレンジング & 当日データの基礎変数化...")
    rename_dict = {}
    for col in df.columns:
        if 'ﾀｲﾑ指数M' in col: rename_dict[col] = 'time_idx_m'
        elif 'ﾀｲﾑ指数' in col: rename_dict[col] = 'time_idx'
        elif 'ｽﾀｰﾄ' in col: rename_dict[col] = 'start_idx'
        elif '追走' in col: rename_dict[col] = 'pace_idx'
        elif '上がり' in col: rename_dict[col] = 'last3f_idx'
        elif '賞金' in col: rename_dict[col] = 'prize'
    df = df.rename(columns=rename_dict)

    df['rank_num'] = pd.to_numeric(df['着順'], errors='coerce')
    df['target_win'] = (df['rank_num'] == 1).astype(int)
    df['target_place'] = (df['rank_num'] <= 3).astype(int)
    
    date_clean = df['date'].astype(str).str.replace('年', '-').str.replace('月', '-').str.replace('日', '').str.replace('/', '-')
    df['date_parsed'] = pd.to_datetime(date_clean, errors='coerce')
    
    for c in ['time_idx', 'time_idx_m', 'start_idx', 'pace_idx', 'last3f_idx']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).replace('**', np.nan), errors='coerce')

    if '馬体重' in df.columns:
        weight_extract = df['馬体重'].astype(str).str.extract(r'(\d+)\s*\(([-+]?\d+)\)')
        df['weight'] = pd.to_numeric(weight_extract[0], errors='coerce')
        df['weight_change'] = pd.to_numeric(weight_extract[1], errors='coerce')
    else:
        df['weight'], df['weight_change'] = np.nan, np.nan

    df['prize'] = pd.to_numeric(df.get('prize', '0').astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    df['trainer'] = df['調教師'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).str.strip() if '調教師' in df.columns else '不明'
    df['race_id_str'] = df['race_id'].astype(str)
    
    places = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    df['place_name'] = df['race_id_str'].str[4:6].map(places).fillna('不明')
    df['is_right_turn'] = df['place_name'].isin(['中山', '阪神', '京都', '小倉', '福島', '札幌', '函館']).astype(int)
    df['is_steep_hill'] = df['place_name'].isin(['中山', '阪神']).astype(int)
    df['kinryo_weight_ratio'] = (df['斤量'] / df['weight'].replace(0, np.nan)).fillna(0.0)
    
    return df

def generate_horse_lags_and_emas(df):
    print("  -> [2/5] 競走馬の過去実績への「ずらし（Shift 1〜5）」・EMA計算...")
    df = df.sort_values(by=['馬名', 'date_parsed', 'race_id'])
    grouped = df.groupby('馬名')
    
    for c in ['rank_num', 'time_idx', 'start_idx', 'pace_idx', 'last3f_idx', 'prize', 'weight', 'weight_change']:
        if c in df.columns:
            for lag in range(1, 6):
                df[f'prev{lag}_{c}'] = grouped[c].shift(lag)

    for c in [c for c in ['time_idx', 'start_idx', 'pace_idx', 'last3f_idx', 'prize'] if c in df.columns]:
        for span in [3, 5, 10, 20]:
            df[f'ema{span}_{c}'] = grouped[c].transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
            
        df[f'trend_3_10_{c}'] = df[f'ema3_{c}'] - df[f'ema10_{c}']
        df[f'trend_5_20_{c}'] = df[f'ema5_{c}'] - df[f'ema20_{c}']
        df[f'max_{c}'] = grouped[c].transform(lambda x: x.shift(1).cummax())
        df[f'ratio_to_max_{c}'] = df[f'prev1_{c}'] / df[f'max_{c}'].replace(0, np.nan)

    df['interval_days'] = grouped['date_parsed'].diff().dt.days
    df['is_long_rest'] = (df['interval_days'] > 180).astype(int)

    df['horse_win_rate'] = grouped['target_win'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    df['horse_place_rate'] = grouped['target_place'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

    return df

def generate_complex_target_encodings_strict(df):
    print("  -> [3/5] 騎手・調教師の「前日までの日付レベル」ターゲットエンコーディング...")
    if '騎手' in df.columns:
        j_daily = df.groupby(['騎手', 'date_parsed']).agg(
            wins=('target_win', 'sum'), places=('target_place', 'sum'), total=('target_win', 'count')
        ).reset_index().sort_values(['騎手', 'date_parsed'])
        
        j_daily['cum_wins'] = j_daily.groupby('騎手')['wins'].cumsum().shift(1).fillna(0)
        j_daily['cum_places'] = j_daily.groupby('騎手')['places'].cumsum().shift(1).fillna(0)
        j_daily['cum_total'] = j_daily.groupby('騎手')['total'].cumsum().shift(1).fillna(0)
        
        j_daily['騎手_win_rate'] = (j_daily['cum_wins'] / j_daily['cum_total'].replace(0, np.nan)).fillna(0.05)
        j_daily['騎手_place_rate'] = (j_daily['cum_places'] / j_daily['cum_total'].replace(0, np.nan)).fillna(0.15)
        
        df = pd.merge(df, j_daily[['騎手', 'date_parsed', '騎手_win_rate', '騎手_place_rate']], on=['騎手', 'date_parsed'], how='left')

        jp_daily = df.groupby(['騎手', 'place_name', 'date_parsed']).agg(
            wins=('target_win', 'sum'), places=('target_place', 'sum'), total=('target_win', 'count')
        ).reset_index().sort_values(['騎手', 'place_name', 'date_parsed'])
        
        jp_daily['cum_wins'] = jp_daily.groupby(['騎手', 'place_name'])['wins'].cumsum().shift(1).fillna(0)
        jp_daily['cum_places'] = jp_daily.groupby(['騎手', 'place_name'])['places'].cumsum().shift(1).fillna(0)
        jp_daily['cum_total'] = jp_daily.groupby(['騎手', 'place_name'])['total'].cumsum().shift(1).fillna(0)
        
        jp_daily['jockey_place_win_rate'] = (jp_daily['cum_wins'] / jp_daily['cum_total'].replace(0, np.nan)).fillna(0.05)
        jp_daily['jockey_place_place_rate'] = (jp_daily['cum_places'] / jp_daily['cum_total'].replace(0, np.nan)).fillna(0.15)
        
        df = pd.merge(df, jp_daily[['騎手', 'place_name', 'date_parsed', 'jockey_place_win_rate', 'jockey_place_place_rate']], on=['騎手', 'place_name', 'date_parsed'], how='left')

    if 'trainer' in df.columns:
        t_daily = df.groupby(['trainer', 'date_parsed']).agg(
            wins=('target_win', 'sum'), places=('target_place', 'sum'), total=('target_win', 'count')
        ).reset_index().sort_values(['trainer', 'date_parsed'])
        
        t_daily['cum_wins'] = t_daily.groupby('trainer')['wins'].cumsum().shift(1).fillna(0)
        t_daily['cum_places'] = t_daily.groupby('trainer')['places'].cumsum().shift(1).fillna(0)
        t_daily['cum_total'] = t_daily.groupby('trainer')['total'].cumsum().shift(1).fillna(0)
        
        t_daily['trainer_win_rate'] = (t_daily['cum_wins'] / t_daily['cum_total'].replace(0, np.nan)).fillna(0.05)
        t_daily['trainer_place_rate'] = (t_daily['cum_places'] / t_daily['cum_total'].replace(0, np.nan)).fillna(0.15)
        
        df = pd.merge(df, t_daily[['trainer', 'date_parsed', 'trainer_win_rate', 'trainer_place_rate']], on=['trainer', 'date_parsed'], how='left')

        tp_daily = df.groupby(['trainer', 'place_name', 'date_parsed']).agg(
            wins=('target_win', 'sum'), total=('target_win', 'count')
        ).reset_index().sort_values(['trainer', 'place_name', 'date_parsed'])
        
        tp_daily['cum_wins'] = tp_daily.groupby(['trainer', 'place_name'])['wins'].cumsum().shift(1).fillna(0)
        tp_daily['cum_total'] = tp_daily.groupby(['trainer', 'place_name'])['total'].cumsum().shift(1).fillna(0)
        
        tp_daily['trainer_place_win_rate'] = (tp_daily['cum_wins'] / tp_daily['cum_total'].replace(0, np.nan)).fillna(0.05)
        
        df = pd.merge(df, tp_daily[['trainer', 'place_name', 'date_parsed', 'trainer_place_win_rate']], on=['trainer', 'place_name', 'date_parsed'], how='left')

    return df

def generate_race_pace_features(df):
    print("  -> [4/5] 過去実績ベースの展開推定...")
    if 'prev1_start_idx' in df.columns:
        sort_cols = [c for c in ['race_id', 'prev1_start_idx'] if c in df.columns]
        df = df.sort_values(by=sort_cols, ascending=[True, False][:len(sort_cols)])
        grouped = df.groupby('race_id')
        
        df['race_expected_pace'] = grouped['prev1_start_idx'].transform(lambda x: x.head(3).mean())
        if 'prev1_last3f_idx' in df.columns:
            df['pace_advantage'] = df['prev1_last3f_idx'] - df['race_expected_pace']
    
    return df.sort_index()

def generate_4d_relative_features(df):
    print("  -> [5/5] ずらした過去特徴量ベースのレース内相対評価4次元展開...")
    sort_cols = [c for c in ['race_id', '馬番'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols)
    
    base_cols = [c for c in df.columns if (c.startswith('prev') or c.startswith('ema') or c.startswith('trend') or c.endswith('_rate') or c in ['kinryo_weight_ratio', 'interval_days', 'weight_change'])]
    valid_cols = [c for c in base_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    grouped = df.groupby('race_id')
    for c in valid_cols:
        c_mean = grouped[c].transform('mean')
        c_std = grouped[c].transform('std').replace(0, 1.0).fillna(1.0)
        c_max = grouped[c].transform('max').replace(0, 1.0)
        
        df[f'{c}_race_diff'] = df[c] - c_mean
        df[f'{c}_race_zscore'] = df[f'{c}_race_diff'] / c_std
        df[f'{c}_race_rank'] = grouped[c].rank(ascending=False, method='min', na_option='bottom')
        df[f'{c}_race_ratio'] = df[c] / c_max

    return df

def main():
    print("==================================================")
    print("🔥 特徴量生成パイプライン実行開始")
    print("==================================================")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ エラー: {INPUT_FILE} が見つかりません。")
        return
        
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    df = clean_data(df)
    df = generate_horse_lags_and_emas(df)
    df = generate_complex_target_encodings_strict(df)
    df = generate_race_pace_features(df)
    df = generate_4d_relative_features(df)
    
    # 💥 計算親となった「当日の生データ列」のみを遮断（学習のCV分割用に 'date' は保持）
    print("💥 計算親となった「当日の生データ列（リーク対象）」を遮断中...")
    raw_leak_cols = [
        '着順', 'target_win', 'target_place',
        'time_idx', 'time_idx_m', 'start_idx', 'pace_idx', 'last3f_idx', 'prize',
        '単勝', '人気', '馬体重', 'weight', 'weight_change',
        'タイム', '着差', '通過', '上り', '調教ﾀｲﾑ', '厩舎ｺﾒﾝﾄ', '備考', '賞金(万円)',
        'ﾀｲﾑ指数\n\n\nタイム指数(通常)\nタイム指数マスター',
        'ﾀｲﾑ指数M\n\n\nタイム指数(通常)\nタイム指数マスター',
        'ｽﾀｰﾄ指数', '追走指数', '上がり指数',
        '馬名', '性齢', '騎手', '調教師', 'trainer', '馬主', 'date_parsed', 'place_name', 'race_id_str'
    ]
    
    df = df.drop(columns=[c for c in raw_leak_cols if c in df.columns], errors='ignore')

    df = reduce_mem_usage(df)
    gc.collect()
    
    print("==================================================")
    print(f"🎉 生成完了！ 最終特徴量数: {df.shape[1]} 列")
    print("==================================================")
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"💾 {OUTPUT_FILE} に保存しました。")

if __name__ == "__main__":
    main()