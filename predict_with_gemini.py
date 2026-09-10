import os
import re
import time
import json
import pandas as pd
import numpy as np
import joblib
import unicodedata
import streamlit as st
from google import genai
from google.genai import types
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from datetime import datetime, timezone, timedelta

# ==========================================
# 🎨 アプリの基本設定 & スタイル定義
# ==========================================
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="🐴", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .section-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 1.2rem 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; white-space: nowrap; font-size: 17px; }
    .kachi-table th { padding: 12px 10px; text-align: center; background: #f8f9fa; border-bottom: 2px solid #dee2e6; font-size: 17px; font-weight: bold; }
    .kachi-table td { padding: 12px 10px; text-align: center; border-bottom: 1px solid #f1f3f5; font-size: 17px; }
    .badge-mark { color: #fff; padding: 6px 12px; border-radius: 6px; font-weight: bold; display: inline-block; min-width: 60px; font-size: 16px; }
    .badge-honmei { background: #e74c3c; } .badge-taikou { background: #3498db; }
    .badge-tana { background: #2ecc71; } .badge-renka { background: #f39c12; }
    .badge-ana { background: #9b59b6; } .badge-keshi { background: #e0e0e0; color: #7f8c8d; }
    
    .sense-card {
        background-color: #ffffff;
        border-left: 6px solid #8e44ad;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .win5-card {
        background-color: #fff9e6;
        border-left: 6px solid #f1c40f;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .win5-title { font-size: 1.2rem; font-weight: bold; color: #d35400; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
    
    .gemini-win5-box {
        background-color: #f8f9fa; border: 2px solid #f1c40f; border-radius: 8px; padding: 20px; margin-top: 10px; line-height: 1.6; font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🐴 AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 

FUTURE_CSV = "future_races.csv"
CACHE_FILE = "app_cache.pkl"

# 🌟 学習時と同じ「ズラし対象（前走データ）」のリストを明記
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

def get_badge_class(mark):
    if pd.isna(mark): return "badge-keshi"
    if "◎" in mark: return "badge-honmei"
    elif "◯" in mark: return "badge-taikou"
    elif "▲" in mark: return "badge-tana"
    elif "△" in mark: return "badge-renka"
    elif "☆" in mark: return "badge-ana"
    return "badge-keshi"

def get_dist_cat(d):
    if pd.isna(d): return np.nan
    if d <= 1400: return 'sprint'
    elif d <= 2200: return 'mile_middle'
    else: return 'stayer'

def parse_weight(val):
    if pd.isna(val): return np.nan, np.nan
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return np.nan, np.nan

class EnsembleModel:
    def __init__(self, lgb_model, xgb_model, cat_model, weights=(0.4, 0.3, 0.3)):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        X_num = X.copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)

        lgb_pred = self.lgb_model.predict(X_num)
        lgb_pred = (lgb_pred - np.mean(lgb_pred)) / (np.std(lgb_pred) + 1e-8)

        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_num))
        xgb_pred = (xgb_pred - np.mean(xgb_pred)) / (np.std(xgb_pred) + 1e-8)

        cat_pred = self.cat_model.predict(X_num)
        cat_pred = (cat_pred - np.mean(cat_pred)) / (np.std(cat_pred) + 1e-8)

        w1, w2, w3 = self.weights
        return w1 * lgb_pred + w2 * xgb_pred + w3 * cat_pred

import __main__
__main__.EnsembleModel = EnsembleModel

@st.cache_resource
def load_model():
    model_paths = ["keiba_ai_model.pkl", "勝ちパカくん.pkl"]
    for m_name in model_paths:
        if os.path.exists(m_name):
            try: 
                return joblib.load(m_name), None
            except Exception as e:
                return None, f"モデルファイル '{m_name}' の読み込みエラー: {e}"
    return None, "モデルファイルが見つかりません。"

@st.cache_resource
def load_encoders():
    le_cond = joblib.load("le_cond.pkl") if os.path.exists("le_cond.pkl") else None
    le_surf = joblib.load("le_surf.pkl") if os.path.exists("le_surf.pkl") else None
    return le_cond, le_surf

@st.cache_data
def load_future_data():
    df_future = pd.DataFrame()
    future_error = None
    if os.path.exists(FUTURE_CSV):
        for enc in ['utf-8-sig', 'utf-8', 'cp932']:
            try:
                df_f = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding=enc)
                if not df_f.empty:
                    df_f['race_id'] = df_f['race_id'].astype(str).str.zfill(12)
                    df_f['place_name'] = df_f['race_id'].str[4:6].map({"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}).fillna("開催場")
                    df_f['r_num'] = pd.to_numeric(df_f['race_id'].str[-2:], errors='coerce').fillna(1).astype(int)
                    df_f['馬名_clean'] = df_f['馬名'].astype(str).apply(clean_horse_name)
                    df_f['day_label'] = df_f['date'].astype(str).str.strip() if 'date' in df_f.columns else "当日"
                    df_f['distance_num'] = pd.to_numeric(df_f.get('distance'), errors='coerce')
                    df_f['dist_cat'] = df_f['distance_num'].apply(get_dist_cat)
                    df_f['kinryo_num'] = pd.to_numeric(df_f.get('斤量'), errors='coerce')
                    df_future = df_f
                    future_error = None
                    break
            except Exception as e:
                future_error = f"'{FUTURE_CSV}' 読み込み失敗 ({enc}): {e}"
    return df_future, future_error

@st.cache_resource
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return joblib.load(CACHE_FILE), None
        except Exception as e:
            return None, f"キャッシュファイル '{CACHE_FILE}' 読み込みエラー: {e}"
    return None, f"キャッシュファイル '{CACHE_FILE}' が見つかりません。"

model_data, model_err = load_model()
le_cond, le_surf = load_encoders()
df_future, future_err = load_future_data()
app_cache, cache_err = load_cache()

if app_cache:
    course_map = app_cache.get('course_map', {})
    past_dict = app_cache.get('past_dict', {})
    trainer_map = app_cache.get('trainer_map', {})
    jockey_track_map = app_cache.get('jockey_track_map', {})
    jockey_map = app_cache.get('jockey_map', {})
else:
    course_map, past_dict, trainer_map, jockey_track_map, jockey_map = {}, {}, {}, {}, {}

def check_paddock_time(time_str):
    if not time_str or ':' not in str(time_str): return False, ""
    try:
        JST = timezone(timedelta(hours=9))
        now = datetime.now(JST)
        
        h, m = map(int, str(time_str).split(':'))
        race_dt = now.replace(hour=h, minute=m, second=0)
        diff_mins = (race_dt - now).total_seconds() / 60
        
        if diff_mins < -12 * 60: diff_mins += 24 * 60
        elif diff_mins > 12 * 60: diff_mins -= 24 * 60
            
        is_close = -15 <= diff_mins <= 60
        msg = f"（発走まで約{int(diff_mins)}分）" if diff_mins >= 0 else f"（発走から約{abs(int(diff_mins))}分経過）"
        return is_close, msg
    except:
        return False, ""
    # ==========================================
# 🧠 AI予測計算メインロジック
# ==========================================
def calculate_predictions(race_id_target, df_fut, cond):
    if df_fut.empty or model_data is None: return None, None, None, None
    race_df = df_fut[df_fut['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None, None, None, None

    model = model_data['model']
    features = model_data.get('features', [])
    
    # 斤量の確実な数値化とデフォルト埋め
    race_df['kinryo_num'] = pd.to_numeric(race_df.get('斤量', race_df.get('kinryo_num')), errors='coerce').fillna(55.0)

    if le_cond is not None and hasattr(le_cond, 'classes_'):
        known_conds = set(le_cond.classes_)
        race_df['condition_code'] = le_cond.transform([cond])[0] if cond in known_conds else 0
    else:
        race_df['condition_code'] = 0

    if le_surf is not None and hasattr(le_surf, 'classes_') and 'surface' in race_df.columns:
        known_surfs = set(le_surf.classes_)
        race_df['surface_code'] = race_df['surface'].apply(lambda x: le_surf.transform([x])[0] if x in known_surfs else 0)

    race_df['race_num'] = pd.to_numeric(race_df['race_id'].astype(str).str[-2:], errors='coerce').fillna(1.0)
    race_df['meet_day_num'] = 1.0
    race_df['track_degradation'] = race_df['meet_day_num'] * race_df['race_num']
    race_df['place_code_str'] = race_df.get('place_code', race_df['race_id'].astype(str).str[4:6]).astype(str)
    race_df['turn_direction'] = race_df['place_code_str'].apply(lambda x: 'left' if str(x) in ['04', '05', '07'] else 'right')
    
    if 'surface' in race_df.columns:
        race_df['course_id'] = race_df['place_code_str'] + "_" + race_df['surface'].astype(str) + "_" + race_df['distance_num'].fillna(0).astype(int).astype(str)
    else:
        race_df['course_id'] = race_df['place_code_str'] + "_芝_" + race_df['distance_num'].fillna(0).astype(int).astype(str)
    
    race_df['course_avg_last3f'] = race_df['course_id'].map(course_map).fillna(50.0)

    target_cols = [
        'last_date', 'prev_dist', 'horse_prize_avg', 'prev_prize', 'prev_1c', 'last_3f_avg_rank', 
        'eff_my_start_idx', 'eff_my_start_idx_long', 'eff_my_last3f_idx', 'eff_my_last3f_idx_long',
        'start_idx_trend', 'last3f_idx_trend', 'last_kinryo', 'prev_last_corner', 'prev_corner_diff', 
        'prev_rank', 'prev2_rank', 'prev_jockey'
    ] + [f'prev_{c}' for c in LEAKY_COLS_TO_SHIFT]

    for col in target_cols:
        race_df[col] = race_df['馬名_clean'].apply(lambda x: past_dict.get(x, {}).get(col, np.nan))

    for f in features:
        if f not in race_df.columns and not f.endswith('_race_diff') and not f.endswith('_race_zscore') and not f.startswith('prev_'):
            race_df[f] = race_df['馬名_clean'].apply(lambda x: past_dict.get(x, {}).get(f, np.nan))

    race_df['prev_1c'] = race_df['prev_1c'].fillna(10.0)
    race_df['prev_last_corner'] = race_df['prev_last_corner'].fillna(10.0)
    race_df['prev_corner_diff'] = race_df['prev_corner_diff'].fillna(0.0)
    race_df['prev_rank'] = race_df['prev_rank'].fillna(7.0)
    race_df['prev2_rank'] = race_df['prev2_rank'].fillna(7.0)
    race_df['eff_my_start_idx'] = race_df['eff_my_start_idx'].fillna(50.0)
    race_df['eff_my_last3f_idx'] = race_df['eff_my_last3f_idx'].fillna(50.0)
    race_df['start_idx_trend'] = race_df['start_idx_trend'].fillna(0.0)
    race_df['last3f_idx_trend'] = race_df['last3f_idx_trend'].fillna(0.0)
    race_df['horse_prize_avg'] = race_df['horse_prize_avg'].fillna(0.0)
    race_df['prev_prize'] = race_df['prev_prize'].fillna(0.0)

    if 'date' in race_df.columns:
        curr_dates = pd.to_datetime(race_df['date'], errors='coerce', utc=True).dt.tz_convert(None).fillna(pd.Timestamp.now())
    else:
        curr_dates = pd.Series([pd.Timestamp.now()] * len(race_df), index=race_df.index)
        
    last_dates = pd.to_datetime(race_df['last_date'], errors='coerce', utc=True).dt.tz_convert(None)
    race_df['interval_days'] = (curr_dates - last_dates).dt.days.fillna(30)
    
    race_df['is_fresh'] = (race_df['interval_days'] >= 60).astype(int)
    race_df['is_tight'] = (race_df['interval_days'] <= 21).astype(int)

    race_df['momentum_rank'] = race_df['prev2_rank'] - race_df['prev_rank']
    race_df['騎手_clean'] = race_df.get('騎手', pd.Series()).astype(str).str.strip()
    race_df['is_jockey_change'] = (race_df['騎手_clean'] != race_df['prev_jockey']).astype(int)

    race_df['race_expected_pace'] = race_df['prev_1c'].mean()

    race_df['dist_change_num'] = race_df['distance_num'] - race_df['prev_dist'].fillna(race_df['distance_num'])
    race_df['same_dist_avg_rank'] = race_df.apply(lambda r: past_dict.get(r['馬名_clean'], {}).get('cat_stats', {}).get(r['dist_cat'], {}).get('avg_rank', 7.0), axis=1)
    race_df['same_place_avg_rank'] = race_df.apply(lambda r: past_dict.get(r['馬名_clean'], {}).get('place_stats', {}).get(r['place_code_str'], 7.0), axis=1)
    race_df['surface_avg_rank'] = race_df.apply(lambda r: past_dict.get(r['馬名_clean'], {}).get('surface_stats', {}).get(r.get('surface', '芝'), 7.0), axis=1)
    race_df['turn_avg_rank'] = race_df.apply(lambda r: past_dict.get(r['馬名_clean'], {}).get('turn_stats', {}).get(r['turn_direction'], 7.0), axis=1)
    
    is_heavy = 'heavy' if cond in ['重', '不良', '稍重'] else 'good'
    race_df['condition_avg_rank'] = race_df.apply(lambda r: past_dict.get(r['馬名_clean'], {}).get('condition_stats', {}).get(is_heavy, 7.0), axis=1)

    race_df['race_avg_prize'] = race_df['horse_prize_avg'].mean()
    if pd.isna(race_df['race_avg_prize'].iloc[0]) or race_df['race_avg_prize'].iloc[0] == 0: 
        race_df['race_avg_prize'] = 1.0
    race_df['race_prize_relative'] = race_df['horse_prize_avg'] / race_df['race_avg_prize']
    race_df['race_prize_rank'] = race_df['horse_prize_avg'].rank(ascending=False, method='min')
    
    race_df['prize_diff_vs_prev'] = race_df['race_avg_prize'] - race_df['prev_prize']

    race_df['first_corner'] = race_df['prev_1c']
    race_df['last_corner'] = race_df['prev_last_corner']
    race_df['corner_diff'] = race_df['prev_corner_diff']

    weight_series = race_df.get('馬体重', pd.Series([np.nan]*len(race_df), index=race_df.index))
    weights_parsed = weight_series.apply(parse_weight)
    race_df['body_weight'] = [p[0] for p in weights_parsed]
    race_df['kinryo_weight_ratio'] = race_df['kinryo_num'] / race_df['body_weight'].fillna(470)
        
    race_df['trainer_win_rate'] = race_df.get('調教師', pd.Series()).map(trainer_map).fillna(0.08)
    race_df['jockey_win_rate'] = race_df['騎手_clean'].map(jockey_map).fillna(0.1).clip(0.0, 1.0)
    
    race_df['jockey_track_win_rate'] = race_df.apply(lambda r: jockey_track_map.get((r.get('騎手_clean'), r.get('place_code_str')), 0.0), axis=1)
    
    base_cols_for_relative = [
        'kinryo_num', 'body_weight', 'interval_days', 
        'eff_my_start_idx', 'eff_my_last3f_idx', 
        'jockey_win_rate', 'trainer_win_rate', 'horse_prize_avg'
    ]
    for col in base_cols_for_relative:
        if col in race_df.columns:
            mean_val = race_df[col].mean()
            std_val = race_df[col].std()
            if pd.isna(std_val) or std_val == 0: std_val = 1.0
            race_df[f'{col}_race_diff'] = race_df[col] - mean_val
            race_df[f'{col}_race_zscore'] = (race_df[col] - mean_val) / std_val

    X = pd.DataFrame(index=race_df.index)
    for f in features:
        X[f] = pd.to_numeric(race_df[f], errors='coerce').fillna(0) if f in race_df.columns else 0.0

    try:
        raw_scores = model.predict(X)
    except Exception as e:
        st.error(f"❌ モデル予測エラー: {e}")
        return None, None, None, None

    min_score, max_score = np.min(raw_scores), np.max(raw_scores)
    if pd.notna(max_score) and pd.notna(min_score) and max_score > min_score:
        norm_scores = (raw_scores - min_score) / (max_score - min_score) * 5.0
        exp_scores = np.exp(norm_scores)
        race_df['win_prob'] = exp_scores / np.sum(exp_scores)
    else:
        race_df['win_prob'] = 1.0 / len(race_df) if len(race_df) > 0 else 0.10

    race_df['top2_prob'] = race_df['win_prob'].apply(lambda p: min(1.0, p * 1.8))
    race_df['top3_prob'] = race_df['win_prob'].apply(lambda p: min(1.0, p * 2.5))

    odds_col = '単勝' if '単勝' in race_df.columns else ('オッズ' if 'オッズ' in race_df.columns else None)
    if odds_col:
        race_df['単勝_num'] = pd.to_numeric(race_df[odds_col].astype(str).str.replace('倍', '').str.replace(',', ''), errors='coerce')
    else:
        race_df['単勝_num'] = np.nan
        
    race_df['単勝_num'] = race_df['単勝_num'].fillna(race_df['単勝_num'].rank(method='min') * 2.5 + 2.0)
    race_df['ev'] = race_df['win_prob'] * race_df['単勝_num']

    if pd.notna(min_score) and max_score > min_score:
        race_df['ai_score'] = (50 + (race_df['win_prob'] - race_df['win_prob'].min()) / (race_df['win_prob'].max() - race_df['win_prob'].min()) * 100).round().astype(int)
    else:
        race_df['ai_score'] = 100

    total_horses = len(race_df)
    if total_horses > 0 and race_df['eff_my_start_idx'].notna().any():
        race_df['start_rank'] = race_df['eff_my_start_idx'].rank(ascending=False, method='min')
        def det_style(row):
            if pd.isna(row.get('start_rank')): return "-" 
            pct = row['start_rank'] / total_horses
            if pct <= 0.15: return "逃げ"
            elif pct <= 0.40: return "先行"
            elif pct <= 0.75: return "差し"
            else: return "追込"
        race_df['脚質'] = race_df.apply(det_style, axis=1)
    else:
        race_df['脚質'] = "-"

    race_df = race_df.sort_values(by='win_prob', ascending=False).reset_index(drop=True)
    race_df['印'] = "消"
    
    marks = ["◎", "◯", "▲", "△", "☆1", "☆2"]
    for i in range(min(len(race_df), len(marks))):
        race_df.loc[i, '印'] = marks[i]

    probs = race_df['win_prob'].values
    p1, p2, p3 = probs[0] if len(probs)>0 else 0, probs[1] if len(probs)>1 else 0, probs[2] if len(probs)>2 else 0
    p4 = probs[3] if len(probs)>3 else 0.05
    
    gap_1_2 = p1 - p2
    gap_1_3 = p1 - p3

    if gap_1_2 >= 0.07 or (gap_1_2 < 0.035 and gap_1_3 >= 0.06):
        pat = "① 1強気配 (軸圧倒)" if gap_1_2 >= 0.07 else "② 2強気配 (頭分け対抗)"
        rec_ticket = "3連単 軸1頭相手4頭マルチ"
        buy_detail = f"1頭軸: {race_df.loc[0, '馬番']}(◎) ⇔ 相手: " + ", ".join([f"{race_df.loc[k, '馬番']}({race_df.loc[k, '印']})" for k in range(1, min(5, len(race_df)))]) if len(race_df)>0 else "-"
    elif (p1 - p4) < 0.08:
        pat = "④ 波乱気配 (大混戦)"
        rec_ticket = "3連複 5頭BOX"
        buy_detail = "BOX: " + ", ".join([f"{race_df.loc[k, '馬番']}({race_df.loc[k, '印']})" for k in range(min(5, len(race_df)))])
    else:
        pat = "③ 混戦気配 (標準展開)"
        rec_ticket = "3連複 ◎1頭軸相手5頭流し"
        buy_detail = f"軸: {race_df.loc[0, '馬番']}(◎) -> 相手: " + ", ".join([f"{race_df.loc[k, '馬番']}({race_df.loc[k, '印']})" for k in range(1, min(6, len(race_df)))]) if len(race_df)>0 else "-"

    return race_df, pat, rec_ticket, buy_detail

# ==========================================
# 📊 HTMLテーブル生成 & UIレンダリング
# ==========================================
def generate_base_table(disp_df, is_newcomer):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>脚質</th><th>斤量</th><th>スコア</th><th>1着率</th><th>3着内率</th><th>オッズ</th><th>期待値</th><th>Python印</th></tr>"
    
    for _, r in disp_df.iterrows():
        ev_val, odds_val = float(r.get('ev', 0)), float(r.get('単勝_num', 0))
        win_val, top3_val = float(r.get('win_prob', 0)), float(r.get('top3_prob', 0))
        mark = r.get('印', '消')
        
        kinryo_val = float(r.get('kinryo_num', 55.0))
        if pd.isna(kinryo_val): kinryo_val = 55.0
        
        score_str = f"<b>{int(r.get('ai_score', 100))}</b>" if not is_newcomer else "-"
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        top3_str = f"<span style='color:#3498db; font-weight:bold;'>{top3_val*100:.1f}%</span>" if top3_val >= 0.50 else f"{top3_val*100:.1f}%"
        ev_str = f"<b>{ev_val:.2f}</b>" if not is_newcomer else "-"

        html += f"<tr>"
        html += f"<td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{r.get('脚質', '-')}</td><td>{kinryo_val:.1f}kg</td><td>{score_str}</td><td>{win_str}</td><td>{top3_str}</td>"
        html += f"<td>{odds_val:.1f}倍</td><td>{ev_str}</td><td><span class='badge-mark {get_badge_class(mark)}'>{mark}</span></td>"
        html += "</tr>"
    html += "</table></div>"
    return html

def generate_fusion_table(merged_df, is_newcomer):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>斤量</th><th>AIｽｺア</th><th>1着率</th><th>期待値</th><th>Python印</th><th>Gemini印</th><th style='text-align:left;'>Gemini短評</th></tr>"
    
    for _, r in merged_df.iterrows():
        win_val = float(r.get('win_prob', 0))
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        
        kinryo_val = float(r.get('kinryo_num', 55.0))
        if pd.isna(kinryo_val): kinryo_val = 55.0
        
        html += f"<tr>"
        html += f"<td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{kinryo_val:.1f}kg</td><td><b>{int(r.get('ai_score', 100))}</b></td><td>{win_str}</td><td><b>{float(r.get('ev', 0)):.2f}</b></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('印', '消'))}'>{r.get('印', '消')}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('Gemini印', '消'))}'>{r.get('Gemini印', '消')}</span></td>"
        html += f"<td style='text-align:left; font-size:15px; color:#444;'>{r.get('短評', '-')}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html

st.sidebar.button("🔄 画面リロード", on_click=lambda: st.cache_data.clear(), use_container_width=True)
st.markdown("<div class='section-header'>🎯 レース選択</div>", unsafe_allow_html=True)

if df_future.empty:
    st.warning("出馬表データがありません。")
    st.stop()

dates = sorted(df_future['day_label'].unique())
sel_date = st.radio("開催日", dates, horizontal=True, label_visibility="collapsed")
day_df = df_future[df_future['day_label'] == sel_date]

with st.expander("👑 今日のWIN5をAIに一発予想させる（Python × Gemini ダブル推奨）"):
    st.write("※Geminiが本日のWIN5対象レースを自動検索し、「Python本命馬」と「Gemini独自推奨馬」の2頭立てで攻略します。")
    if st.button("🔥 WIN5の買い目を生成する", type="primary", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("【設定エラー】APIキーが見つかりません。")
        else:
            with st.spinner("Geminiが本日のWIN5対象レースを検索し、買い目を構築中...（数分かかります）"):
                main_races = day_df[day_df['r_num'].isin([9, 10, 11, 12])].sort_values(by=['r_num', 'place_name'])
                win5_prompt_text = ""
                
                for r_id in main_races['race_id'].unique():
                    r_rows = main_races[main_races['race_id'] == r_id]
                    if r_rows.empty: continue
                    p_name = r_rows.iloc[0]['place_name']
                    r_n = r_rows.iloc[0]['r_num']
                    rc_name = r_rows.iloc[0].get('race_name', '')
                    
                    r_df, _, _, _ = calculate_predictions(r_id, df_future, "良")
                    if r_df is not None and not r_df.empty:
                        python_top = f"{int(r_df.iloc[0]['馬番'])}番 {r_df.iloc[0]['馬名']}"
                        all_horses = [f"{int(r['馬番'])}番 {r['馬名']}" for _, r in r_df.iterrows()]
                        
                        win5_prompt_text += f"■ {p_name} {r_n}R 【{rc_name}】\n"
                        win5_prompt_text += f"  🤖 Python本命: {python_top}\n"
                        win5_prompt_text += f"  出走馬: {', '.join(all_horses)}\n\n"
                
                win5_sys_prompt = f"""
                あなたは超一流のWIN5予想職人（Gemini）です。
                以下のデータは、本日の各競馬場のメインレース付近のAI予測スコアです。
                
                【最重要ミッション：対象レースの特定】
                まずは必ず「JRA {sel_date} WIN5 対象レース」でGoogle検索を行い、本日の【公式のWIN5対象の5レース】を正確に特定してください。（勝手に推測して選ぶのは厳禁です）
                
                【予想ミッション】
                1. 検索で特定した【本物の対象5レース】についてのみ、以下のダブル推奨形式で予想を展開してください。
                2. 各レースの「🤖 Python本命（定量データ1位）」は既に決まっています。
                3. あなたはGoogle検索を駆使して定性データ（陣営コメント、直近の気配など）を独自に調べ、Pythonに引きずられない【🧠 Gemini独立推奨馬】を各レース1〜2頭ピックアップしてください。
                
                【出力フォーマット】
                ### 📍 [競馬場] [レース番号]R 【レース名】 (WIN5 対象Xレース目)
                *   🤖 **Python推奨**: [Python本命馬] (定量データトップ)
                *   🧠 **Gemini推奨**: [あなたが独自に選んだ馬]
                *   📝 **見解**: [なぜその馬をGemini枠として推奨するのか、展開予想など]
                """
                
                try:
                    client = genai.Client(api_key=GEMINI_API_KEY)
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"【本日のWIN5候補レース群】\n{win5_prompt_text}",
                        config=types.GenerateContentConfig(
                            system_instruction=win5_sys_prompt,
                            temperature=0.3,
                            tools=[{"googleSearch": {}}]
                        )
                    )
                    
                    st.markdown(f"""
                    <div class='gemini-win5-box'>
                        {response.text}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")

places = day_df['place_name'].unique()
place_tabs = st.tabs([f"📍 {p}" for p in places])
for i, place in enumerate(places):
    with place_tabs[i]:
        place_df = day_df[day_df['place_name'] == place]
        races = sorted(place_df['r_num'].unique())
        for j in range(0, len(races), 6):
            cols = st.columns(6)
            for k in range(6):
                if j + k < len(races):
                    r = races[j + k]
                    r_df = place_df[place_df['r_num'] == r]
                    r_id = r_df['race_id'].iloc[0]
                    r_name = str(r_df.get('race_name', pd.Series([''])).iloc[0]).strip()
                    if cols[k].button(f"{r}R {r_name}" if r_name and r_name!='nan' else f"{r}R", key=f"btn_{r_id}", use_container_width=True):
                        st.session_state['selected_race_id'] = r_id

if st.session_state['selected_race_id']:
    t_id = str(st.session_state['selected_race_id'])
    t_rows = df_future[df_future['race_id'].astype(str) == t_id]
    if t_rows.empty: st.stop()

    r_info = t_rows.iloc[0]
    r_name = r_info.get('race_name', '')
    is_newcomer = "新馬" in str(r_name)
    
    st.markdown("---")
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{r_name}】</h2>", unsafe_allow_html=True)
    
    cond = st.radio("想定馬場 (AI予測に反映されます)", ["良", "稍重", "重", "不良"], horizontal=True)
    
    res_df, pat, rec_ticket, buy_detail = calculate_predictions(t_id, df_future, cond)
    
    if res_df is not None:
        st.markdown(f"""
        <div class='sense-card'>
            <div class='sense-title'>🧠 AI勝負気配判定: {pat}</div>
            <div style='margin-top: 8px;'><b>🎟️ 推奨買い目:</b> <span class='ticket-badge'>{rec_ticket}</span></div>
            <div style='margin-top: 6px; font-size: 15px; color: #555;'><b>📝 買い目詳細:</b> {buy_detail}</div>
        </div>
        """, unsafe_allow_html=True)

        sort_option = st.selectbox("🔄 テーブルの表示順", ["システム推奨（印順）", "馬番順（昇順）", "AIスコア順", "予想オッズ順"], index=0)
        if sort_option == "馬番順（昇順）": res_df = res_df.sort_values(by='馬番')
        elif sort_option == "AIスコア順": res_df = res_df.sort_values(by=['ai_score'], ascending=False)
        elif sort_option == "予想オッズ順": res_df = res_df.sort_values(by=['単勝_num'])
        
        table_placeholder = st.empty()
        with table_placeholder.container():
            st.markdown("<div class='section-header'>📊 Pythonベース予想データ</div>", unsafe_allow_html=True)
            st.markdown(generate_base_table(res_df, is_newcomer), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        race_time_str = ""
        for col_name in ['発走', '発走時刻', '発走時間', 'time']:
            if col_name in r_info and pd.notna(r_info[col_name]) and str(r_info[col_name]).strip() != "":
                race_time_str = str(r_info[col_name]).strip()
                match = re.search(r'\d{1,2}:\d{2}', race_time_str)
                if match:
                    race_time_str = match.group(0)
                break
                
        is_paddock_close, time_msg = check_paddock_time(race_time_str)
        
        display_time = race_time_str if race_time_str else "不明（手動でパドック検索をONにできます）"
        st.info(f"🕒 発走予定時刻: {display_time} {time_msg}")
        use_paddock = st.checkbox("🐎 レース直前：Geminiに「パドック・馬体気配」を最優先で検索させる", value=is_paddock_close)

        if st.button("🧠 Geminiを独立させて、泥臭く穴馬を発掘させる", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            table_summary = []
            for _, r in res_df.iterrows():
                curr_kinryo = r.get('kinryo_num', 55.0)
                last_kinryo = r.get('last_kinryo', curr_kinryo)
                if pd.isna(curr_kinryo): curr_kinryo = 55.0
                if pd.isna(last_kinryo): last_kinryo = curr_kinryo
                
                kinryo_diff = curr_kinryo - last_kinryo
                diff_str = f"{kinryo_diff:+.1f}kg" if kinryo_diff != 0 else "±0kg"
                
                info = f"馬番:{int(r.get('馬番',0)):02d} | 馬名:{r.get('馬名','')} | 脚質:{r.get('脚質','')} | 斤量:{curr_kinryo:.1f}kg (前走比 {diff_str})"
                table_summary.append(info)

            paddock_instruction = ""
            if use_paddock:
                paddock_instruction = """
【特別直前ミッション：最重要】
このレースは発走直前です。必ず「馬名 パドック」「馬名 馬体重 X(Twitter)」などで検索し、
現在のリアルタイムのパドック状態、テンション、馬体増やイレ込みなどの「直前気配」を最優先で評価の根拠にしてください。
"""
            system_instruction = f"""
あなたはプロ競馬予想家（トラックマン）です。
【あなたのワークフロー】
1. 提供された「出走馬リスト」を確認してください。今回、各馬の「斤量」と「前走からの斤量増減」も提供されています。
2. Google検索ツールを駆使して、以下の【定性情報】を最優先で検索・収集してください。
   ①【調教・勝負気配】: 今回はメイチ（本気）か、次を見据えた叩き台か。最終追い切りの動き。
   ②【血統・馬場適性】: 今日のコースや馬場状態に対する血統的な裏付け。
   ③【斤量増減の影響】: 今回の斤量（および前走からの増減）が、この馬の過去実績や評価に対してプラスかマイナスか。
   ④【前走不利】: 前走大敗だが今回巻き返せる「隠れた実力馬」の発掘。
{paddock_instruction}
3. 検索で得た定性情報のみを基に、最終評価を下してください。

【印の打ち方】
◎(1頭), ◯(1頭), ▲(1頭), △(1頭), ☆(1〜2頭), 消(それ以外)
※テキスト解説不要。必ず以下のJSONのみ出力すること。短評には検索で得た定性情報（調教、血統、斤量、気配など）を具体的に書くこと。
{{ "evaluations": [ {{"馬番": 1, "Gemini印": "◎", "短評": "〇〇のため好走必至"}}, ... ] }}
"""
            
            prompt = f"対象レース: {sel_date} {r_info['place_name']} {r_info['r_num']}R 【{r_name}】\n【今日の想定馬場状態】: {cond}\n【出走馬リスト】:\n{chr(10).join(table_summary)}"

            with st.spinner("🧠 GeminiがPythonに頼らず、独自の視点で検索中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                gemini_data = None
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash', 
                            contents=prompt, 
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction, temperature=0.7, tools=[{"googleSearch": {}}]
                            )
                        )
                        res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                        if res_text:
                            match = re.search(r'\{.*\}', res_text, re.DOTALL)
                            gemini_data = json.loads(match.group(0) if match else res_text)
                            break
                    except Exception as e:
                        if attempt < 2: time.sleep(3)
                        else: st.error(f"❌ APIエラー: {e}")
                    
                if gemini_data:
                    evals = gemini_data.get("evaluations", [])
                    eval_df = pd.DataFrame(evals)
                    if not eval_df.empty and '馬番' in eval_df.columns:
                        eval_df['馬番_num'] = pd.to_numeric(eval_df['馬番'], errors='coerce').astype('Int64')
                        eval_df_clean = eval_df[['馬番_num', 'Gemini印', '短評']].dropna(subset=['馬番_num'])
                        
                        res_df['馬番_num'] = pd.to_numeric(res_df['馬番'], errors='coerce').astype('Int64')
                        merged_df = pd.merge(res_df, eval_df_clean, on='馬番_num', how='left')
                        merged_df['Gemini印'] = merged_df['Gemini印'].fillna('消')
                        merged_df['短評'] = merged_df['短評'].fillna('-')
                        
                        table_placeholder.empty()
                        with table_placeholder.container():
                            st.markdown("<div class='section-header'>🔥 Python（確率脳） × Gemini（定性脳） 独立評価比較</div>", unsafe_allow_html=True)
                            st.markdown(generate_fusion_table(merged_df, is_newcomer), unsafe_allow_html=True)