import pandas as pd
import numpy as np
import joblib
import unicodedata
import re

ML_TARGET_CSV = "ml_target_data_v2.csv"

# ズラし対象リスト
LEAKY_COLS_TO_SHIFT = [
    'first_half_time', 'first_pos', 'jp_runs', 'jp_wins', '人気', '単勝',
    'horse_runs', 'horse_wins', 'horse_win_rate',
    'jockey_runs', 'jockey_wins', 'jockey_win_power',
    'my_time_idx', 'my_last3f_idx', 'my_pace_idx', 'my_start_idx', 'prize_num',
    'time_sec_clean', 'last3f_sec_clean', 'first_pos_clean', 'last_pos_clean', 
    'first_corner', 'last_corner', 'corner_diff',
    'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx',
    'hybrid_power_idx', 'pace_scenario_idx', 'race_avg_start_idx', 
    'course_avg_time', 'course_avg_first'
]

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    s = re.sub(r'[\s・･.\-ー_]+', '', s).strip()
    return s.upper()

def get_dist_cat(d):
    if pd.isna(d): return np.nan
    if d <= 1400: return 'sprint'
    elif d <= 2200: return 'mile_middle'
    else: return 'stayer'

print("🔄 過去データを読み込み中...")
df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding='utf-8-sig')
df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)
df_past['date_parsed'] = pd.to_datetime(df_past['date'], errors='coerce')
df_past['distance_num'] = pd.to_numeric(df_past.get('distance'), errors='coerce')
df_past['dist_cat'] = df_past['distance_num'].apply(get_dist_cat)
df_past['rank_num'] = pd.to_numeric(df_past.get('着順'), errors='coerce')
df_past['is_win_past'] = (df_past['rank_num'] == 1).astype(int)

place_code = df_past.get('place_code', pd.Series(['00']*len(df_past)))
surface = df_past.get('surface', pd.Series(['芝']*len(df_past)))
df_past['course_id'] = place_code.astype(str) + "_" + surface.astype(str) + "_" + df_past['distance_num'].fillna(0).astype(int).astype(str)
df_past['place_code_str'] = df_past.get('place_code', df_past['race_id'].astype(str).str[4:6]).astype(str)
df_past = df_past.sort_values(by='date_parsed')

print("🧠 コース辞書を作成中...")
c_map = {}
if 'course_id' in df_past.columns:
    for c_id, group in df_past.groupby('course_id'):
        v = group['course_avg_last3f'].dropna() if 'course_avg_last3f' in group.columns else pd.Series(dtype=float)
        c_map[c_id] = v.iloc[-1] if len(v) > 0 else 50.0

print("🐴 馬・騎手・調教師の辞書を作成中...")
horse_dict = {}
df_past['騎手_clean'] = df_past['騎手'].astype(str).str.strip()
jockey_map = df_past.dropna(subset=['rank_num']).groupby('騎手_clean')['is_win_past'].mean().to_dict()

for horse, group in df_past.groupby('馬名_clean'):
    valid_past = group.dropna(subset=['rank_num']).copy()
    if valid_past.empty: 
        last_valid_row = group.iloc[-1]
        prev2_rank = 7.0
        prev_jockey = ""
    else: 
        last_valid_row = valid_past.iloc[-1]
        if len(valid_past) >= 2:
            prev2_rank = valid_past.iloc[-2]['rank_num']
        else:
            prev2_rank = 7.0
        prev_jockey = str(last_valid_row.get('騎手_clean', "")).strip()
    
    def parse_pass_full(val):
        if pd.isna(val): return np.nan, np.nan, np.nan
        parts = str(val).split('-')
        try: return float(parts[0]), float(parts[-1]), float(parts[0]) - float(parts[-1])
        except: return np.nan, np.nan, np.nan
    
    p_1c, p_lc, p_cdiff = parse_pass_full(last_valid_row.get('通過', np.nan))
    
    prize_col = '賞金(万円)' if '賞金(万円)' in valid_past.columns else 'prize'
    prizes = pd.to_numeric(valid_past.get(prize_col, pd.Series()), errors='coerce').fillna(0)
    horse_prize_avg = prizes.ewm(span=5, min_periods=1).mean().iloc[-1] if not prizes.empty else 0.0
    
    cat_stats = {}
    for cat_name in ['sprint', 'mile_middle', 'stayer']:
        c_rows = valid_past[valid_past['dist_cat'] == cat_name]
        cat_stats[cat_name] = {'avg_rank': c_rows['rank_num'].ewm(span=3, min_periods=1).mean().iloc[-1] if len(c_rows) > 0 else 7.0}
        
    place_stats = {}
    for p_code in valid_past['place_code'].astype(str).unique():
        p_rows = valid_past[valid_past['place_code'].astype(str) == p_code]
        place_stats[p_code] = p_rows['rank_num'].ewm(span=3, min_periods=1).mean().iloc[-1] if len(p_rows) > 0 else 7.0 
        
    surface_stats = {}
    for surf in valid_past['surface'].astype(str).unique():
        s_rows = valid_past[valid_past['surface'].astype(str) == surf]
        surface_stats[surf] = s_rows['rank_num'].ewm(span=3, min_periods=1).mean().iloc[-1] if len(s_rows) > 0 else 7.0
        
    turn_stats = {}
    valid_past['turn_direction'] = valid_past['place_code_str'].apply(lambda x: 'left' if str(x) in ['04', '05', '07'] else 'right')
    for turn in ['left', 'right']:
        t_rows = valid_past[valid_past['turn_direction'] == turn]
        turn_stats[turn] = t_rows['rank_num'].ewm(span=3, min_periods=1).mean().iloc[-1] if len(t_rows) > 0 else 7.0

    condition_stats = {}
    valid_past['is_heavy_track'] = valid_past['condition'].astype(str).apply(lambda x: 'heavy' if x in ['重', '不良', '稍重'] else 'good')
    for cond_key in ['good', 'heavy']:
        c_rows = valid_past[valid_past['is_heavy_track'] == cond_key]
        condition_stats[cond_key] = c_rows['rank_num'].ewm(span=3, min_periods=1).mean().iloc[-1] if len(c_rows) > 0 else 7.0

    s_vals = pd.to_numeric(valid_past.get('my_start_idx', pd.Series()), errors='coerce').dropna()
    eff_my_start_idx = s_vals.ewm(span=3, min_periods=1).mean().iloc[-1] if not s_vals.empty else 50.0
    eff_my_start_idx_long = s_vals.ewm(span=10, min_periods=1).mean().iloc[-1] if not s_vals.empty else 50.0

    l_vals = pd.to_numeric(valid_past.get('my_last3f_idx', pd.Series()), errors='coerce').dropna()
    eff_my_last3f_idx = l_vals.ewm(span=3, min_periods=1).mean().iloc[-1] if not l_vals.empty else 50.0
    eff_my_last3f_idx_long = l_vals.ewm(span=10, min_periods=1).mean().iloc[-1] if not l_vals.empty else 50.0

    start_idx_trend = eff_my_start_idx - eff_my_start_idx_long
    last3f_idx_trend = eff_my_last3f_idx - eff_my_last3f_idx_long

    prev_prize = pd.to_numeric(last_valid_row.get(prize_col, 0.0), errors='coerce')
    prev_prize = prev_prize if pd.notna(prev_prize) else 0.0

    horse_data = {
        'last_date': last_valid_row.get('date_parsed', np.nan),
        'last_kinryo': pd.to_numeric(last_valid_row.get('kinryo_num', 55.0), errors='coerce'),
        'prev_dist': pd.to_numeric(last_valid_row.get('distance_num', np.nan), errors='coerce'), 
        'horse_prize_avg': horse_prize_avg, 
        'prev_prize': prev_prize,
        'prev_1c': p_1c if not pd.isna(p_1c) else 10.0, 
        'prev_last_corner': p_lc if not pd.isna(p_lc) else 10.0,
        'prev_corner_diff': p_cdiff if not pd.isna(p_cdiff) else 0.0,
        'prev_rank': last_valid_row.get('rank_num', 7.0),
        'prev2_rank': prev2_rank,
        'prev_jockey': prev_jockey,
        'cat_stats': cat_stats,
        'place_stats': place_stats,
        'surface_stats': surface_stats,
        'turn_stats': turn_stats,
        'condition_stats': condition_stats,
        'eff_my_start_idx': eff_my_start_idx,
        'eff_my_start_idx_long': eff_my_start_idx_long,
        'eff_my_last3f_idx': eff_my_last3f_idx,
        'eff_my_last3f_idx_long': eff_my_last3f_idx_long,
        'start_idx_trend': start_idx_trend,
        'last3f_idx_trend': last3f_idx_trend,
        'last_3f_avg_rank': eff_my_last3f_idx
    }
    
    if pd.isna(horse_data['last_kinryo']):
        horse_data['last_kinryo'] = 55.0

    for col in LEAKY_COLS_TO_SHIFT:
        horse_data[f'prev_{col}'] = last_valid_row.get(col, np.nan)

    for col in last_valid_row.index:
        if col not in horse_data:
            horse_data[col] = last_valid_row.get(col, np.nan)

    horse_dict[horse] = horse_data

trainer_map = df_past.groupby('調教師')['is_win_past'].mean().to_dict()
jockey_track_map = df_past.dropna(subset=['rank_num']).groupby(['騎手_clean', 'place_code_str'])['is_win_past'].mean().to_dict()

print("💾 キャッシュファイル（app_cache.pkl）を保存中...")
joblib.dump({
    'course_map': c_map,
    'past_dict': horse_dict,
    'trainer_map': trainer_map,
    'jockey_track_map': jockey_track_map,
    'jockey_map': jockey_map
}, 'app_cache.pkl')

print("✅ キャッシュの作成が完了しました！このファイルをアプリにアップロードしてください。")