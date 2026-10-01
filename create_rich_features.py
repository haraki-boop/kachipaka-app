import os
import re
import gc
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = 'keiba_database2.csv'
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
    print("  -> [1/5] クレンジング & プレミアムデータの基礎変数化...")
    
    df['rank_num'] = pd.to_numeric(df['着順'], errors='coerce')
    df['target_win'] = (df['rank_num'] == 1).astype(int)
    df['target_place'] = (df['rank_num'] <= 3).astype(int)
    
    date_clean = df['date'].astype(str).str.replace('年', '-').str.replace('月', '-').str.replace('日', '').str.replace('/', '-')
    df['date_parsed'] = pd.to_datetime(date_clean, errors='coerce')
    
    num_cols = ['タイム指数', '調子偏差値', '同条件_馬場指数', '同距離_馬場指数', '他距離_馬場指数', '単勝オッズ', '通過', '上り']
    for c in num_cols:
        if c in df.columns:
            if c == '通過':
                df[c] = pd.to_numeric(df[c].astype(str).str.split('-').str[0], errors='coerce')
            else:
                df[c] = pd.to_numeric(df[c].astype(str).replace('**', np.nan).replace('', np.nan), errors='coerce')

    if '馬体重' in df.columns:
        weight_extract = df['馬体重'].astype(str).str.extract(r'(\d+)\s*\(([-+]?\d+)\)')
        df['weight'] = pd.to_numeric(weight_extract[0], errors='coerce')
        df['weight_change'] = pd.to_numeric(weight_extract[1], errors='coerce')
    else:
        df['weight'], df['weight_change'] = np.nan, np.nan

    df['prize'] = pd.to_numeric(df.get('賞金(万円)', '0').astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
    df['trainer'] = df['調教師'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).str.strip() if '調教師' in df.columns else '不明'
    
    df['race_id_str'] = df['race_id'].astype(str)
    places = {'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}
    df['place_name'] = df['race_id_str'].str[4:6].map(places).fillna('不明')
    df['is_right_turn'] = df['place_name'].isin(['中山', '阪神', '京都', '小倉', '福島', '札幌', '函館']).astype(int)
    df['is_steep_hill'] = df['place_name'].isin(['中山', '阪神']).astype(int)
    
    # 馬場と距離のカテゴリ化
    df['surface'] = df['surface'].fillna('不明')
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
    df['dist_category'] = pd.cut(df['distance'], bins=[0, 1300, 1700, 2100, 4000], labels=['短距離', 'マイル', '中距離', '長距離']).astype(str)
    
    df['kinryo_weight_ratio'] = (pd.to_numeric(df['斤量'], errors='coerce') / df['weight'].replace(0, np.nan)).fillna(0.0)
    
    return df

def generate_horse_lags_and_emas(df):
    print("  -> [2/5] 競走馬の過去実績フル展開（Shift 1〜5 / EMA / 移動標準偏差）...")
    df = df.sort_values(by=['馬名', 'date_parsed', 'race_id'])
    grouped = df.groupby('馬名')
    
    # 5走前までフル拡張
    shift_targets = ['rank_num', 'タイム指数', '調子偏差値', '同条件_馬場指数', '同距離_馬場指数', '他距離_馬場指数', 'prize', 'weight', 'weight_change', '通過', '上り']
    for c in shift_targets:
        if c in df.columns:
            for lag in range(1, 6): 
                df[f'prev{lag}_{c}'] = grouped[c].shift(lag)

    # EMA 3, 5, 10, 20 + 安定度(標準偏差)
    ema_targets = ['タイム指数', '調子偏差値', 'prize', '通過', '上り']
    for c in [x for x in ema_targets if x in df.columns]:
        for span in [3, 5, 10, 20]:
            df[f'ema{span}_{c}'] = grouped[c].transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
            # 過去X戦のムラ（標準偏差）を計算して安定度を測る
            df[f'std{span}_{c}'] = grouped[c].transform(lambda x: x.shift(1).rolling(window=span, min_periods=1).std()).fillna(0)
            
        df[f'trend_3_10_{c}'] = df[f'ema3_{c}'] - df[f'ema10_{c}']
        df[f'trend_5_20_{c}'] = df[f'ema5_{c}'] - df[f'ema20_{c}']
        df[f'max_{c}'] = grouped[c].transform(lambda x: x.shift(1).cummax())
        df[f'ratio_to_max_{c}'] = df[f'prev1_{c}'] / df[f'max_{c}'].replace(0, np.nan)

    df['interval_days'] = grouped['date_parsed'].diff().dt.days
    df['is_long_rest'] = (df['interval_days'] > 180).astype(int)
    
    if '調子偏差値' in df.columns:
        df['is_dev_up_after_rest'] = ((df['is_long_rest'] == 1) & (df['調子偏差値'] > df['prev1_調子偏差値'])).astype(int)

    df['horse_win_rate'] = grouped['target_win'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
    df['horse_place_rate'] = grouped['target_place'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)

    return df

def generate_target_encodings(df, group_cols, prefix):
    """汎用ターゲットエンコーディング関数"""
    daily = df.groupby(group_cols + ['date_parsed']).agg(
        wins=('target_win', 'sum'), places=('target_place', 'sum'), total=('target_win', 'count')
    ).reset_index().sort_values(group_cols + ['date_parsed'])
    
    daily['cum_wins'] = daily.groupby(group_cols)['wins'].cumsum().shift(1).fillna(0)
    daily['cum_places'] = daily.groupby(group_cols)['places'].cumsum().shift(1).fillna(0)
    daily['cum_total'] = daily.groupby(group_cols)['total'].cumsum().shift(1).fillna(0)
    
    daily[f'{prefix}_win_rate'] = (daily['cum_wins'] / daily['cum_total'].replace(0, np.nan)).fillna(0.05)
    daily[f'{prefix}_place_rate'] = (daily['cum_places'] / daily['cum_total'].replace(0, np.nan)).fillna(0.15)
    
    merge_cols = group_cols + ['date_parsed', f'{prefix}_win_rate', f'{prefix}_place_rate']
    return pd.merge(df, daily[merge_cols], on=group_cols + ['date_parsed'], how='left')

def generate_nlp_target_encodings(df):
    print("  -> [3/5] カテゴリ変数（条件別フルコンボ・調教評価）のターゲットエンコーディング...")
    
    if '騎手' in df.columns:
        df = generate_target_encodings(df, ['騎手'], 'jockey')
        df = generate_target_encodings(df, ['騎手', 'place_name'], 'jockey_place')
        df = generate_target_encodings(df, ['騎手', 'surface'], 'jockey_surface') # 芝ダート別
        df = generate_target_encodings(df, ['騎手', 'dist_category'], 'jockey_dist') # 距離別

    if 'trainer' in df.columns:
        df = generate_target_encodings(df, ['trainer'], 'trainer')
        df = generate_target_encodings(df, ['trainer', 'place_name'], 'trainer_place')
        df = generate_target_encodings(df, ['trainer', 'surface'], 'trainer_surface')
        
    if '騎手' in df.columns and 'trainer' in df.columns:
        # 騎手 × 調教師の黄金コンビ勝率
        df = generate_target_encodings(df, ['騎手', 'trainer'], 'jockey_trainer_combo')

    if '調教評価' in df.columns:
        eval_map = {'A': 3, 'B': 2, 'C': 1, 'S': 4} 
        df['train_eval_num'] = df['調教評価'].map(eval_map).fillna(0)

    if '厩舎コメント' in df.columns:
        positive_words = ['先着', '併入', '馬也', '先行']
        negative_words = ['遅れ', '一杯']
        df['comment_pos'] = df['厩舎コメント'].fillna('').apply(lambda x: sum([1 for w in positive_words if w in str(x)]))
        df['comment_neg'] = df['厩舎コメント'].fillna('').apply(lambda x: sum([1 for w in negative_words if w in str(x)]))
        df['comment_score'] = df['comment_pos'] - df['comment_neg']

    return df

def generate_race_pace_features(df):
    print("  -> [4/5] 展開推定（脚質の組み合わせ）...")
    if '脚質' in df.columns:
        df['temp_is_nige'] = (df['脚質'] == '逃').astype(int)
        df['race_nige_count'] = df.groupby('race_id')['temp_is_nige'].transform('sum')
        df['pace_high_advantage'] = ((df['race_nige_count'] >= 3) & (df['脚質'].isin(['差', '追']))).astype(int)
        df = df.drop(columns=['temp_is_nige'])
        
    if 'prev1_通過' in df.columns:
        df['race_expected_pace'] = df.groupby('race_id')['prev1_通過'].transform(lambda x: x.nsmallest(3).mean())
    
    return df.sort_index()

def generate_4d_relative_features(df):
    print("  -> [5/5] 4次元展開（レース内偏差値・ランク）を全生成特徴量に適用...")
    sort_cols = [c for c in ['race_id', '馬番'] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols)
    
    # 全ての生成された数値列を相対評価の対象にする（一気に数百列増殖）
    base_cols = [c for c in df.columns if (
        c.startswith('prev') or c.startswith('ema') or c.startswith('std') or 
        c.startswith('trend') or c.endswith('_rate') or 
        c in ['kinryo_weight_ratio', 'interval_days', '調子偏差値', '同条件_馬場指数', '同距離_馬場指数', '他距離_馬場指数', 'train_eval_num', 'comment_score']
    )]
    valid_cols = [c for c in base_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    grouped = df.groupby('race_id')
    for c in valid_cols:
        c_mean = grouped[c].transform('mean')
        c_std = grouped[c].transform('std').replace(0, 1.0).fillna(1.0)
        c_max = grouped[c].transform('max').replace(0, 1.0)
        
        # メモリ節約のためfloat32で計算
        df[f'{c}_race_diff'] = (df[c] - c_mean).astype(np.float32)
        df[f'{c}_race_zscore'] = (df[f'{c}_race_diff'] / c_std).astype(np.float32)
        df[f'{c}_race_rank'] = grouped[c].rank(ascending=False, method='min', na_option='bottom').astype(np.float32)
        df[f'{c}_race_ratio'] = (df[c] / c_max).astype(np.float32)

    return df

def main():
    print("==================================================")
    print("🔥 限界突破！全次元解放 特徴量生成パイプライン実行開始")
    print("==================================================")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ エラー: {INPUT_FILE} が見つかりません。")
        return
        
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, low_memory=False, encoding='cp932')
    
    df = clean_data(df)
    df = generate_horse_lags_and_emas(df)
    df = generate_nlp_target_encodings(df)
    df = generate_race_pace_features(df)
    df = generate_4d_relative_features(df)
    
    print("💥 当日の生データ列（リーク対象）を遮断中...")
    raw_leak_cols = [
        '着順', 'target_win', 'target_place', 'prize',
        '単勝オッズ', '人気', '馬体重', 'weight', 'weight_change',
        'タイム', '着差', '通過', '上り', '賞金(万円)',
        'タイム指数', '調子偏差値', '同条件_馬場指数', '同距離_馬場指数', '他距離_馬場指数',
        '調教師', 'trainer', '馬主', 'place_name', 'race_id_str', 'dist_category', 'surface',
        '調教タイム', '調教短評', '厩舎コメント' 
    ]
    
    df = df.drop(columns=[c for c in raw_leak_cols if c in df.columns], errors='ignore')

    df = reduce_mem_usage(df)
    gc.collect()
    
    print("==================================================")
    print(f"🎉 生成完了！ 最終特徴量数: {df.shape[1]} 列（Optunaチューニング準備完了）")
    print("==================================================")
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"💾 {OUTPUT_FILE} に保存しました。")

if __name__ == "__main__":
    main()