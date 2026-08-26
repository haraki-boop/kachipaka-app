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

# ==========================================
# 🎨 アプリの基本設定
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
    
    /* 勝負気配カードのスタイル */
    .sense-card {
        background-color: #ffffff;
        border-left: 6px solid #8e44ad;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("🐴 AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 

FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data_v2.csv"

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

def parse_sex_age(val):
    if pd.isna(val): return 0, 4.0
    val = str(val).strip()
    sex_char = val[0] if len(val) > 0 else '牡'
    sex_code = 0 if sex_char == '牡' else (1 if sex_char == '牝' else 2)
    try: age = float(val[1:])
    except: age = 4.0
    return sex_code, age

def parse_weight(val):
    if pd.isna(val): return np.nan, np.nan
    s = str(val).strip()
    m = re.match(r'(\d+)(?:\(([-+]?\d+)\))?', s)
    if m:
        w = float(m.group(1))
        diff = float(m.group(2)) if m.group(2) else 0.0
        return w, diff
    return np.nan, np.nan

# ==========================================
# 🌟 NEW: ダミーのEnsembleModelクラス（Pickle読み込み用）
# ==========================================
class EnsembleModel:
    def __init__(self, lgb_model, xgb_model, cat_model, weights=(0.4, 0.3, 0.3)):
        self.lgb_model = lgb_model
        self.xgb_model = xgb_model
        self.cat_model = cat_model
        self.weights = weights

    def predict(self, X):
        import xgboost as xgb
        X_num = X.copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)

        # LightGBM
        lgb_pred = self.lgb_model.predict(X_num)
        lgb_std = np.std(lgb_pred)
        lgb_norm = (lgb_pred - np.mean(lgb_pred)) / (lgb_std + 1e-5) if lgb_std > 0 else lgb_pred

        # XGBoost
        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_num))
        xgb_std = np.std(xgb_pred)
        xgb_norm = (xgb_pred - np.mean(xgb_pred)) / (xgb_std + 1e-5) if xgb_std > 0 else xgb_pred

        # CatBoost
        cat_pred = self.cat_model.predict(X_num)
        cat_std = np.std(cat_pred)
        cat_norm = (cat_pred - np.mean(cat_pred)) / (cat_std + 1e-5) if cat_std > 0 else cat_pred

        w1, w2, w3 = self.weights
        return w1 * lgb_norm + w2 * xgb_norm + w3 * cat_norm

# Pythonのメインスコープにクラスを登録しておかないと読み込みエラーになるため
import __main__
__main__.EnsembleModel = EnsembleModel

# ==========================================
# 1. データとモデルの読み込み
# ==========================================
@st.cache_resource
def load_model():
    model_paths = ["keiba_ai_model.pkl", "勝ちパカくん.pkl"]
    for m_name in model_paths:
        if os.path.exists(m_name):
            try: 
                return joblib.load(m_name), None
            except Exception as e:
                return None, f"モデルファイル '{m_name}' の読み込みエラー: {e}"
    return None, f"モデルファイル ({', '.join(model_paths)}) が見つかりません。"

@st.cache_data
def load_data():
    df_past = pd.DataFrame()
    past_error = None
    if not os.path.exists(ML_TARGET_CSV):
        past_error = f"過去データファイル '{ML_TARGET_CSV}' が存在しません。"
    else:
        for enc in ['utf-8-sig', 'utf-8', 'cp932']:
            try:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding=enc)
                df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)
                df_past['date_parsed'] = pd.to_datetime(df_past['date'], errors='coerce')
                df_past['distance_num'] = pd.to_numeric(df_past.get('distance'), errors='coerce')
                df_past['dist_cat'] = df_past['distance_num'].apply(get_dist_cat)
                df_past['rank_num'] = pd.to_numeric(df_past.get('着順'), errors='coerce')
                df_past['is_win_past'] = (df_past['rank_num'] == 1).astype(int)
                df_past['kinryo_num'] = pd.to_numeric(df_past.get('斤量'), errors='coerce')
                df_past['wakuban_num'] = pd.to_numeric(df_past.get('枠番'), errors='coerce')
                
                place_code = df_past.get('place_code', pd.Series(['00']*len(df_past)))
                surface = df_past.get('surface', pd.Series(['芝']*len(df_past)))
                df_past['course_id'] = place_code.astype(str) + "_" + surface.astype(str) + "_" + df_past['distance_num'].fillna(0).astype(int).astype(str)
                df_past['course_frame_id'] = df_past['course_id'] + "_frame_" + df_past['wakuban_num'].fillna(0).astype(int).astype(str)
                df_past = df_past.sort_values(by='date_parsed')
                past_error = None
                break
            except Exception as e:
                past_error = f"'{ML_TARGET_CSV}' の読み込み失敗 ({enc}): {e}"

    df_future = pd.DataFrame()
    future_error = None
    if not os.path.exists(FUTURE_CSV):
        future_error = f"出馬表データファイル '{FUTURE_CSV}' が存在しません。"
    else:
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
                    df_f['wakuban_num'] = pd.to_numeric(df_f.get('枠番'), errors='coerce')
                    df_f['umaban_num'] = pd.to_numeric(df_f.get('馬番'), errors='coerce')
                    
                    place_code_f = df_f.get('place_code', df_f['race_id'].str[4:6])
                    surface_f = df_f.get('surface', pd.Series(['芝']*len(df_f)))
                    df_f['course_id'] = place_code_f.astype(str) + "_" + surface_f.astype(str) + "_" + df_f['distance_num'].fillna(0).astype(int).astype(str)
                    df_f['course_frame_id'] = df_f['course_id'] + "_frame_" + df_f['wakuban_num'].fillna(0).astype(int).astype(str)
                    df_future = df_f
                    future_error = None
                    break
            except Exception as e:
                future_error = f"'{FUTURE_CSV}' の読み込み失敗 ({enc}): {e}"

    return df_past, df_future, past_error, future_error

model_data, model_err = load_model()
df_past, df_future, past_err, future_err = load_data()

if model_err:
    st.error(f"❌ 【モデルエラー】 {model_err}")
    st.stop()

if future_err:
    st.error(f"❌ 【出馬表データエラー】 {future_err}")
    st.stop()

if past_err:
    st.warning(f"⚠️ 【過去データ警告】 {past_err}")

# ==========================================
# 2. 過去データ辞書化
# ==========================================
@st.cache_data
def build_past_horse_dict(df_p):
    if df_p.empty: return {}, {}, {}, {}, {}, {}, {}
    horse_dict = {}
    for horse, group in df_p.groupby('馬名_clean'):
        if not horse: continue
        last_row = group.iloc[-1]
        
        t_vals = pd.to_numeric(group.get('my_time_idx', pd.Series()), errors='coerce').clip(40, 80).dropna()
        l_vals = pd.to_numeric(group.get('my_last3f_idx', pd.Series()), errors='coerce').clip(40, 80).dropna()
        p_vals = pd.to_numeric(group.get('my_pace_idx', pd.Series()), errors='coerce').clip(40, 80).dropna()
        s_vals = pd.to_numeric(group.get('my_start_idx', pd.Series()), errors='coerce').clip(40, 80).dropna()
        ranks = pd.to_numeric(group.get('rank_num', pd.Series()), errors='coerce').dropna()

        top3_rows = group[group['rank_num'] <= 3]
        best_dist_avg = top3_rows['distance_num'].mean() if not top3_rows.empty else np.nan

        cat_stats = {}
        for cat_name in ['sprint', 'mile_middle', 'stayer']:
            c_rows = group[group['dist_cat'] == cat_name]
            c_runs = len(c_rows)
            c_wins = (c_rows['rank_num'] == 1).sum()
            cat_stats[cat_name] = {
                'runs': c_runs,
                'win_rate': (c_wins / c_runs) if c_runs > 0 else np.nan
            }

        horse_dict[horse] = {
            'last_date': last_row['date_parsed'],
            'last_jockey': str(last_row.get('騎手', '')).strip(),
            'last_kinryo': last_row['kinryo_num'],
            'prev_prize': pd.to_numeric(last_row.get('賞金(万円)'), errors='coerce'),
            'prev_rank_num': last_row['rank_num'],
            'best_dist_avg': best_dist_avg,
            'cat_stats': cat_stats,
            'eff_rank_avg': ranks.tail(3).mean() if not ranks.empty else 8.0,
            'eff_top5_rate': (ranks.tail(5) <= 5).mean() if not ranks.empty else 0.2,
            'eff_top3_rate': (ranks.tail(5) <= 3).mean() if not ranks.empty else 0.1,
            'eff_my_time_idx': t_vals.tail(3).median() if not t_vals.empty else 50.0,
            'eff_my_last3f_idx': l_vals.tail(3).median() if not l_vals.empty else 50.0,
            'eff_my_pace_idx': p_vals.tail(3).median() if not p_vals.empty else 50.0,
            'eff_my_start_idx': s_vals.tail(3).median() if not s_vals.empty else 50.0,
            'horse_runs': pd.to_numeric(last_row.get('horse_runs'), errors='coerce'),
            'horse_wins': pd.to_numeric(last_row.get('horse_wins'), errors='coerce'),
            'horse_win_rate': pd.to_numeric(last_row.get('horse_win_rate'), errors='coerce'),
            'horse_avg_time_idx': pd.to_numeric(last_row.get('horse_avg_time_idx'), errors='coerce'),
            'horse_avg_last3f_idx': pd.to_numeric(last_row.get('horse_avg_last3f_idx'), errors='coerce'),
            'horse_avg_pace_idx': pd.to_numeric(last_row.get('horse_avg_pace_idx'), errors='coerce'),
            'horse_avg_start_idx': pd.to_numeric(last_row.get('horse_avg_start_idx'), errors='coerce')
        }

    course_front_map = df_p.groupby('course_id')['rank_num'].apply(lambda x: (x <= 3).mean()).to_dict()
    course_frame_map = df_p.groupby('course_frame_id')['rank_num'].apply(lambda x: (x == 1).mean()).to_dict()
    
    trainer_map = df_p.groupby('調教師')['trainer_win_rate'].last().to_dict() if 'trainer_win_rate' in df_p.columns else df_p.groupby('調教師')['is_win_past'].mean().to_dict()
    df_p['騎手_clean'] = df_p['騎手'].astype(str).str.strip()
    combo_map = df_p.groupby(['調教師', '騎手_clean'])['trainer_jockey_combo'].last().to_dict() if 'trainer_jockey_combo' in df_p.columns else df_p.groupby(['調教師', '騎手_clean'])['is_win_past'].mean().to_dict()
    horse_track_map = df_p.groupby(['馬名_clean', 'place_code'])['horse_track_win_rate'].last().to_dict() if 'horse_track_win_rate' in df_p.columns else df_p.groupby(['馬名_clean', 'place_code'])['is_win_past'].mean().to_dict()
    frame_map = df_p.groupby(['place_code', 'distance_num', 'wakuban_num'])['frame_win_rate'].last().to_dict() if 'frame_win_rate' in df_p.columns else df_p.groupby(['place_code', 'distance_num', 'wakuban_num'])['is_win_past'].mean().to_dict()

    return horse_dict, course_front_map, course_frame_map, trainer_map, combo_map, horse_track_map, frame_map

past_dict, course_front_map, course_frame_map, trainer_map, combo_map, horse_track_map, frame_map = build_past_horse_dict(df_past)

# ==========================================
# 3. AI推論＆勝負気配算出ロジック
# ==========================================
def calculate_predictions(race_id_target, df_fut, cond):
    if df_fut.empty or model_data is None: return None, None, None, None
    race_df = df_fut[df_fut['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None, None, None, None

    model = model_data['model']
    features = model_data['features']

    def safe_numeric_col(df, col_name, default_val):
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors='coerce').fillna(default_val)
        return pd.Series(default_val, index=df.index)

    race_df['race_num'] = race_df['race_id'].astype(str).str[-2:]
    race_df['race_num'] = pd.to_numeric(race_df['race_num'], errors='coerce').fillna(1.0)
    race_df['meet_day_num'] = safe_numeric_col(race_df, 'meet_day_num', 1.0)
    race_df['track_degradation'] = race_df['meet_day_num'] * race_df['race_num']

    target_cols = [
        'last_date', 'prev_prize', 'prev_rank_num', 
        'eff_rank_avg', 'eff_top5_rate', 'eff_top3_rate',
        'eff_my_time_idx', 'eff_my_last3f_idx',
        'eff_my_pace_idx', 'eff_my_start_idx',
        'horse_runs', 'horse_wins', 'horse_win_rate',
        'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx'
    ]
    for col in target_cols:
        race_df[col] = race_df['馬名_clean'].apply(lambda x: past_dict.get(x, {}).get(col, np.nan))

    if 'eff_my_start_idx' in race_df.columns and 'eff_my_last3f_idx' in race_df.columns:
        race_df['race_avg_start_idx'] = race_df['eff_my_start_idx'].mean()
        race_df['pace_scenario_idx'] = race_df['eff_my_last3f_idx'].fillna(50.0) * (race_df['race_avg_start_idx'] / 50.0)

    if '性齢' in race_df.columns:
        sex_age = race_df['性齢'].apply(parse_sex_age)
    else:
        sex_age = pd.Series([(0, 4.0)] * len(race_df), index=race_df.index)
        
    race_df['sex_code'] = [s[0] for s in sex_age]
    race_df['age'] = [s[1] for s in sex_age]

    if '馬体重' in race_df.columns:
        weights_parsed = race_df['馬体重'].apply(parse_weight)
    else:
        weights_parsed = pd.Series([(np.nan, np.nan)] * len(race_df), index=race_df.index)
        
    race_df['body_weight'] = [p[0] for p in weights_parsed]
    race_df['body_weight_diff'] = [p[1] for p in weights_parsed]
    race_df['kinryo_body_ratio'] = race_df['kinryo_num'] / race_df['body_weight'].fillna(470)

    if 'place_code' in race_df.columns:
        race_df['place_code_str'] = race_df['place_code'].astype(str)
    else:
        race_df['place_code_str'] = race_df['race_id'].astype(str).str[4:6]
        
    if '調教師' in race_df.columns:
        race_df['trainer_win_rate'] = race_df['調教師'].map(trainer_map).fillna(0.08)
    else:
        race_df['trainer_win_rate'] = 0.08
        
    if '騎手' in race_df.columns:
        race_df['騎手_clean'] = race_df['騎手'].astype(str).str.strip()
    else:
        race_df['騎手_clean'] = ''
        
    race_df['trainer_jockey_combo'] = race_df.apply(lambda r: combo_map.get((r.get('調教師'), r.get('騎手_clean')), 0.08), axis=1)
    race_df['horse_track_win_rate'] = race_df.apply(lambda r: horse_track_map.get((r.get('馬名_clean'), r.get('place_code_str')), 0.0), axis=1)
    race_df['frame_win_rate'] = race_df.apply(lambda r: frame_map.get((r.get('place_code_str'), r.get('distance_num'), r.get('wakuban_num')), 0.08), axis=1)

    def calc_kinryo_diff(row):
        p_kinryo = past_dict.get(row['馬名_clean'], {}).get('last_kinryo', np.nan)
        c_kinryo = row['kinryo_num']
        return c_kinryo - p_kinryo if pd.notna(c_kinryo) and pd.notna(p_kinryo) else 0.0

    def calc_is_same_jockey(row):
        p_jockey = past_dict.get(row['馬名_clean'], {}).get('last_jockey', '')
        c_jockey = str(row.get('騎手', '')).strip()
        return 1 if (c_jockey and c_jockey == p_jockey) else 0

    def calc_dist_diff(row):
        b_avg = past_dict.get(row['馬名_clean'], {}).get('best_dist_avg', np.nan)
        c_dist = row['distance_num']
        return abs(c_dist - b_avg) if pd.notna(b_avg) and pd.notna(c_dist) else 0.0

    def calc_cat_win_rate(row):
        return past_dict.get(row['馬名_clean'], {}).get('cat_stats', {}).get(row['dist_cat'], {}).get('win_rate', np.nan)

    def calc_cat_runs(row):
        return past_dict.get(row['馬名_clean'], {}).get('cat_stats', {}).get(row['dist_cat'], {}).get('runs', 0)

    race_df['kinryo_diff'] = race_df.apply(calc_kinryo_diff, axis=1)
    race_df['is_same_jockey'] = race_df.apply(calc_is_same_jockey, axis=1)
    race_df['dist_diff'] = race_df.apply(calc_dist_diff, axis=1)
    race_df['cat_win_rate'] = race_df.apply(calc_cat_win_rate, axis=1)
    race_df['cat_runs'] = race_df.apply(calc_cat_runs, axis=1)

    race_df['course_frame_win_rate'] = race_df['course_frame_id'].map(course_frame_map).fillna(0.08)
    race_df['course_front_rate'] = race_df['course_id'].map(course_front_map).fillna(0.3)
    race_df['style_course_fit'] = race_df['eff_my_start_idx'].fillna(50) * race_df['course_front_rate']

    if 'date' in race_df.columns:
        race_df['date_parsed_fut'] = pd.to_datetime(race_df['date'], errors='coerce')
    else:
        race_df['date_parsed_fut'] = pd.to_datetime('today')
        
    race_df['interval_days'] = (race_df['date_parsed_fut'] - race_df['last_date']).dt.days.fillna(30)
    race_df['is_long_rest'] = (race_df['interval_days'] >= 180).astype(int)

    if 'jockey_win_power' in race_df.columns:
        j_col = race_df['jockey_win_power']
    elif 'jockey_win_rate' in race_df.columns:
        j_col = race_df['jockey_win_rate']
    else:
        j_col = pd.Series(0.1, index=race_df.index)
        
    race_df['jockey_win_rate'] = pd.to_numeric(j_col, errors='coerce').fillna(0.1).clip(0.0, 1.0)
    race_df['jockey_track_win_rate'] = safe_numeric_col(race_df, 'jockey_track_win_rate', 0.1).clip(0.0, 1.0)
    race_df['jockey_runs'] = safe_numeric_col(race_df, 'jockey_runs', 100)
    race_df['jockey_wins'] = safe_numeric_col(race_df, 'jockey_wins', 10)
    race_df['jp_runs'] = safe_numeric_col(race_df, 'jp_runs', 20)
    race_df['jp_wins'] = safe_numeric_col(race_df, 'jp_wins', 2)
    race_df['jp_win_rate'] = race_df['jp_wins'] / (race_df['jp_runs'] + 1e-5)

    race_df['condition_code'] = cond
    race_df['condition'] = cond

    race_df['time_idx_diff'] = race_df['eff_my_time_idx'] - race_df['horse_avg_time_idx']
    race_df['last3f_idx_diff'] = race_df['eff_my_last3f_idx'] - race_df['horse_avg_last3f_idx']
    race_df['pace_idx_diff'] = race_df['eff_my_pace_idx'] - race_df['horse_avg_pace_idx']
    race_df['start_idx_diff'] = race_df['eff_my_start_idx'] - race_df['horse_avg_start_idx']

    race_df['first_corner_pos'] = safe_numeric_col(race_df, 'first_pos', 8.0)
    race_df['last_corner_pos'] = race_df['first_corner_pos']
    race_df['pos_gain'] = 0.0
    
    if 'eff_my_time_idx' in race_df.columns and 'eff_my_start_idx' in race_df.columns:
        race_df['eff_hybrid_power_idx'] = race_df['eff_my_time_idx'] * 0.5 + race_df['eff_my_start_idx'] * 0.5

    z_cols = [
        'eff_rank_avg', 'eff_top3_rate', 'prev_prize', 'eff_my_time_idx', 
        'eff_my_last3f_idx', 'eff_my_pace_idx', 'jockey_win_rate', 'jockey_track_win_rate',
        'horse_win_rate', 'horse_avg_time_idx', 'horse_avg_last3f_idx', 'horse_avg_pace_idx', 'horse_avg_start_idx',
        'age', 'kinryo_num', 'body_weight', 'prev_rank', 'first_pos',
        'eff_hybrid_power_idx', 'pace_scenario_idx'
    ]
    for c in z_cols:
        if c in race_df.columns:
            s_val = pd.to_numeric(race_df[c], errors='coerce')
            s_std = s_val.std()
            if pd.isna(s_std) or s_std == 0:
                race_df[f'{c}_z'] = 0.0
            else:
                race_df[f'{c}_z'] = (s_val - s_val.mean()) / s_std

    rank_cols = ['eff_my_time_idx', 'horse_avg_time_idx', 'jockey_win_rate', 'horse_win_rate', 'eff_hybrid_power_idx']
    for c in rank_cols:
        if c in race_df.columns:
            s_val = pd.to_numeric(race_df[c], errors='coerce')
            race_df[f'{c}_rank_in_race'] = s_val.rank(ascending=False, method='min')

    X = pd.DataFrame(index=race_df.index)
    for f in features:
        X[f] = race_df[f] if f in race_df.columns else np.nan

    cat_cols = ['surface_code', 'condition_code', 'sex_code']
    for cat in cat_cols:
        if cat in X.columns:
            if cat == 'sex_code': 
                X[cat] = X[cat].replace({'牡':0,'牝':1,'セ':2}).fillna(0)
            elif cat == 'surface_code': 
                X[cat] = X[cat].replace({'芝':0,'ダート':1,'障害':2}).fillna(0)
            elif cat == 'condition_code': 
                X[cat] = X[cat].replace({'良':0,'稍重':1,'稍':1,'重':2,'不良':3}).fillna(0)

    X_num = X.copy()
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)

    try:
        raw_scores = model.predict(X_num)
    except Exception as e:
        st.error(f"❌ モデル予測エラー: {e}")
        return None, None, None, None

    s_std = np.std(raw_scores)
    if pd.notna(s_std) and s_std > 0:
        z_scores = (raw_scores - np.mean(raw_scores)) / s_std
        base_probs = 1.0 / (1.0 + np.exp(-1.2 * z_scores))
        race_df['win_prob'] = base_probs * 0.35 + 0.01
    else:
        race_df['win_prob'] = 0.10

    if '単勝' in race_df.columns:
        race_df['単勝_num'] = pd.to_numeric(race_df['単勝'].astype(str).str.replace('倍', ''), errors='coerce').fillna(10.0)
    elif 'オッズ' in race_df.columns:
        race_df['単勝_num'] = pd.to_numeric(race_df['オッズ'].astype(str).str.replace('倍', ''), errors='coerce').fillna(10.0)
    else:
        race_df['単勝_num'] = 10.0
        
    race_df['人気'] = race_df['単勝_num'].rank(method='min')
    race_df['ev'] = race_df['win_prob'] * race_df['単勝_num']

    p_min = race_df['win_prob'].min()
    p_max = race_df['win_prob'].max()
    if pd.notna(p_min) and pd.notna(p_max) and p_max > p_min:
        race_df['ai_score'] = (50 + (race_df['win_prob'] - p_min) / (p_max - p_min) * 100).round().astype(int)
    else:
        race_df['ai_score'] = 100

    total_horses = len(race_df)
    
    if total_horses > 0 and race_df['eff_my_start_idx'].notna().any():
        if race_df['eff_my_start_idx'].std() == 0:
            race_df['脚質'] = "-"
        else:
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

    race_df = race_df.sort_values(by=['ai_score', 'win_prob'], ascending=[False, False]).reset_index(drop=True)
    race_df['印'] = "消"
    
    if len(race_df) > 0: race_df.loc[0, '印'] = "◎"
    if len(race_df) > 1: race_df.loc[1, '印'] = "◯"
    if len(race_df) > 2: race_df.loc[2, '印'] = "▲"
    if len(race_df) > 3: race_df.loc[3, '印'] = "△"
    if len(race_df) > 4: race_df.loc[4, '印'] = "☆1"
    if len(race_df) > 5: race_df.loc[5, '印'] = "☆2"

    mark_order = {"◎":1, "◯":2, "▲":3, "△":4, "☆1":5, "☆2":6, "消":7}
    race_df['mark_rank'] = race_df['印'].map(mark_order).fillna(99)
    race_df = race_df.sort_values(by='mark_rank').reset_index(drop=True)

    probs = race_df['win_prob'].values
    p1, p2, p3, p4 = probs[0], probs[1], probs[2], probs[3] if len(probs)>3 else 0.05
    gap_1_2 = p1 - p2
    gap_1_3 = p1 - p3

    is_3ren_tan_race = (gap_1_2 >= 0.07) or (gap_1_2 < 0.035 and gap_1_3 >= 0.06)

    if is_3ren_tan_race:
        pat = "① 1強気配 (軸圧倒)" if gap_1_2 >= 0.07 else "② 2強気配 (頭分け対抗)"
        rec_ticket = "3連単 軸1頭相手4頭マルチ (24点 / 2,400円)"
        u_aite = [f"{race_df.loc[k, '馬番']}({race_df.loc[k, '印']})" for k in range(1, min(5, len(race_df)))]
        buy_detail = f"1頭軸: {race_df.loc[0, '馬番']}(◎) ⇔ 相手: {', '.join(u_aite)}"
    elif (p1 - p4) < 0.08:
        pat = "④ 波乱気配 (大混戦)"
        rec_ticket = "3連複 5頭BOX (10点 / 1,000円)"
        u_list = [f"{race_df.loc[k, '馬番']}({race_df.loc[k, '印']})" for k in range(min(5, len(race_df)))]
        buy_detail = f"BOX: {', '.join(u_list)}"
    else:
        pat = "③ 混戦気配 (標準展開)"
        rec_ticket = "3連複 ◎1頭軸相手5頭流し (10点 / 1,000円)"
        u_others = [f"{race_df.loc[k, '馬番']}({race_df.loc[k, '印']})" for k in range(1, min(6, len(race_df)))]
        buy_detail = f"軸: {race_df.loc[0, '馬番']}(◎) -> 相手: {', '.join(u_others)}"

    return race_df, pat, rec_ticket, buy_detail

# ==========================================
# 4. テーブル生成
# ==========================================
def generate_base_table(disp_df, is_newcomer):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>脚質</th><th>AIスコア</th><th>AI勝率</th><th>オッズ</th><th>期待値</th><th>Python印</th></tr>"
    
    for _, r in disp_df.iterrows():
        ev_val = float(r.get('ev', 0)) if pd.notna(r.get('ev')) else 0.0
        odds_val = float(r.get('単勝_num', 0)) if pd.notna(r.get('単勝_num')) else 0.0
        win_prob_val = float(r.get('win_prob', 0)) if pd.notna(r.get('win_prob')) else 0.0
        mark = r.get('印', '消')
        b_cls = get_badge_class(mark)
        
        score_str = f"<b>{int(r.get('ai_score', 100))}</b>" if not is_newcomer else "-"
        win_str = f"{win_prob_val*100:.1f}%" if win_prob_val > 0 else "-"
        ev_str = f"<b>{ev_val:.2f}</b>" if ev_val > 0 and not is_newcomer else "-"
        
        html += f"<tr>"
        html += f"<td><b>{int(r['馬番']):02d}</b></td>"
        html += f"<td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{r.get('脚質', '-')}</td>"
        html += f"<td>{score_str}</td>"
        html += f"<td>{win_str}</td>"
        html += f"<td>{odds_val:.1f}倍</td>"
        html += f"<td>{ev_str}</td>"
        html += f"<td><span class='badge-mark {b_cls}'>{mark}</span></td>"
        html += f"</tr>"
    html += "</table></div>"
    return html

def generate_fusion_table(merged_df, is_newcomer):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>AIｽｺア</th><th>AI勝率</th><th>期待値</th><th>Python印</th><th>Gemini印</th><th style='text-align:left;'>Gemini短評</th></tr>"
    
    for _, r in merged_df.iterrows():
        sys_mark = r.get('印', '消') 
        gem_mark = r.get('Gemini印', '消')
        sys_cls = get_badge_class(sys_mark)
        gem_cls = get_badge_class(gem_mark)
        
        score_str = f"<b>{int(r.get('ai_score', 100))}</b>" if not is_newcomer else "-"
        win_prob_val = float(r.get('win_prob', 0)) if pd.notna(r.get('win_prob')) else 0.0
        win_str = f"{win_prob_val*100:.1f}%" if win_prob_val > 0 else "-"
        ev_str = f"<b>{float(r.get('ev', 0)):.2f}</b>" if float(r.get('ev', 0)) > 0 and not is_newcomer else "-"
        
        html += f"<tr>"
        html += f"<td><b>{int(r['馬番']):02d}</b></td>"
        html += f"<td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{score_str}</td>"
        html += f"<td>{win_str}</td>"
        html += f"<td>{ev_str}</td>"
        html += f"<td><span class='badge-mark {sys_cls}'>{sys_mark}</span></td>"
        html += f"<td><span class='badge-mark {gem_cls}'>{gem_mark}</span></td>"
        html += f"<td style='text-align:left; font-size:15px; color:#444;'>{r.get('短評', '-')}</td>"
        html += f"</tr>"
    html += "</table></div>"
    return html

# ==========================================
# 5. メインUI
# ==========================================
st.sidebar.button("🔄 画面リロード", on_click=lambda: st.cache_data.clear(), use_container_width=True)

st.markdown("<div class='section-header'>🎯 レース選択</div>", unsafe_allow_html=True)
dates = sorted(df_future['day_label'].unique())
sel_date = st.radio("開催日", dates, horizontal=True, label_visibility="collapsed")
day_df = df_future[df_future['day_label'] == sel_date]
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
                    label = f"{r}R {r_name}" if r_name and r_name != 'nan' else f"{r}R"
                    if cols[k].button(label, key=f"btn_{r_id}", use_container_width=True):
                        st.session_state['selected_race_id'] = r_id

if st.session_state['selected_race_id']:
    t_id = str(st.session_state['selected_race_id'])
    t_rows = df_future[df_future['race_id'].astype(str) == t_id]
    if t_rows.empty: st.rerun()

    r_info = t_rows.iloc[0]
    r_name = r_info.get('race_name', '')
    is_newcomer = "新馬" in str(r_name)
    
    st.markdown("---")
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{r_name}】</h2>", unsafe_allow_html=True)
    
    cond = st.radio("想定馬場", ["良", "稍重", "重", "不良"], horizontal=True)
    
    res_df, pat, rec_ticket, buy_detail = calculate_predictions(t_id, df_future, cond)
    
    if res_df is not None:
        st.markdown(f"""
        <div class='sense-card'>
            <div class='sense-title'>🧠 AI勝負気配判定: {pat}</div>
            <div style='margin-top: 8px;'>
                <b>🎟️ 推奨買い目:</b> <span class='ticket-badge'>{rec_ticket}</span>
            </div>
            <div style='margin-top: 6px; font-size: 15px; color: #555;'>
                <b>📝 買い目詳細:</b> {buy_detail}
            </div>
        </div>
        """, unsafe_allow_html=True)

        sort_option = st.selectbox("🔄 テーブルの表示順", ["システム推奨（印順）", "馬番順（昇順）", "AIスコア順（高い順）", "勝率順（高い順）", "予想オッズ順（低い順）", "期待値順（高い順）"], index=0)

        if sort_option == "馬番順（昇順）": res_df = res_df.sort_values(by='馬番')
        elif sort_option == "AIスコア順（高い順）": res_df = res_df.sort_values(by=['ai_score', '馬番'], ascending=[False, True])
        elif sort_option == "勝率順（高い順）": res_df = res_df.sort_values(by=['win_prob', '馬番'], ascending=[False, True])
        elif sort_option == "予想オッズ順（低い順）": res_df = res_df.sort_values(by=['単勝_num', '馬番'], ascending=[True, True])
        elif sort_option == "期待値順（高い順）": res_df = res_df.sort_values(by=['ev', '馬番'], ascending=[False, True])
        
        table_placeholder = st.empty()
        with table_placeholder.container():
            st.markdown("<div class='section-header'>📊 Pythonベース予想データ</div>", unsafe_allow_html=True)
            if is_newcomer: st.info("🐣 新馬戦のため、過去データが存在せずベースAIスコアは参考値です。")
            st.markdown(generate_base_table(res_df, is_newcomer), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 調教・血統・敗因データを検索し、表を最強アップデートする", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
            if res_df is None or len(res_df) < 6:
                st.error("データが不足しているため予想をスキップします。")
                st.stop()

            table_summary = []
            for idx, row in res_df.iterrows():
                ev_val = float(row.get('ev', 0)) if pd.notna(row.get('ev')) else 0.0
                odds_val = float(row.get('単勝_num', 0)) if pd.notna(row.get('単勝_num')) else 0.0
                win_val = float(row.get('win_prob', 0)) * 100 if pd.notna(row.get('win_prob')) else 0.0
                table_summary.append(f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | 脚質:{row.get('脚質', '不明')} | オッズ:{odds_val}倍 | AI勝率:{win_val:.1f}% | 期待値:{ev_val:.2f} | Python印:{row.get('印', '消')}")

            system_instruction = f"""
あなたはプロの競馬分析AI「勝ちぱかくん」の最終意思決定者（Gemini脳）です。
Pythonが算出した定量データ（ベース予測）に、あなたが検索で得る「最新の定性データ」を掛け合わせ、最終的な評価を下す（骨格を太くする）のがあなたの役割です。

【あなたの3大定性チェック・ワークフロー】
1. まず、提供された「Python算出データ」を確認してください。
2. 次に、Google検索ツールを駆使して、各出走馬に関する以下の【3大定性情報】を深掘り検索してください。
   ①【調教（追い切り状態）】: 前走時と比較した状態の変化、勝負気配
   ②【血統（コース/馬場適性）】: 今日のコース・馬場状態に対する血統的裏付け
   ③【前走敗因・不利】: 前走の大敗に明確な不利や理由があり、今回巻き返せるか
   ※注意: 検索のタイムアウトを防ぐため、Python印が上位の馬や、期待値が高い穴馬を中心に重点的に検索・評価を行ってください。
3. Pythonの定量データと、あなたが検索で得た定性情報を掛け合わせ、あなた自身の最終評価（Gemini印と短評）を下してください。

【重要：カンニング絶対禁止ルール】
実際のレース結果（着順・配当など）を検索してカンニングすることは絶対に禁止です。発走前の事前情報のみで評価を構成してください。

【出力フォーマット】
テキストによる解説は一切不要です。必ず以下のJSON形式のみで出力してください（マークダウンの ```json などは絶対に入れないでください）。

{{
  "evaluations": [
    {{"馬番": 1, "Gemini印": "◎", "短評": "坂路自己ベスト。洋芝適性高く好勝負"}},
    {{"馬番": 2, "Gemini印": "☆", "短評": "前走直線詰まる。巻き返し期待"}}
  ]
}}
"""
            prompt = f"対象レース: {sel_date} {r_info['place_name']} {r_info['r_num']}R 【{r_name}】\n【想定馬場状態】: {cond}\n【AI勝負気配】: {pat} ({rec_ticket})\n\n【Python算定データ】:\n{chr(10).join(table_summary)}"

            with st.spinner("🧠 Geminiが調教・血統・敗因データを検索＆分析し、表をアップデート中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                gemini_data = None
                
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash', 
                            contents=prompt, 
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction, 
                                temperature=0.2,
                                tools=[{"googleSearch": {}}],
                                
                            )
                        )
                        res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                        
                        if res_text:
                            match = re.search(r'\{.*\}', res_text, re.DOTALL)
                            clean_json_text = match.group(0) if match else res_text
                            gemini_data = json.loads(clean_json_text)
                            break
                            
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(3)
                        else:
                            st.error(f"❌ API通信または解析エラー: {e}")
                    
                if gemini_data:
                    try:
                        evals = gemini_data.get("evaluations", [])
                        eval_df = pd.DataFrame(evals)
                        if not eval_df.empty and '馬番' in eval_df.columns:
                            eval_df['馬番_num'] = pd.to_numeric(eval_df['馬番'], errors='coerce')
                            if 'Gemini印' not in eval_df.columns: eval_df['Gemini印'] = '消'
                            if '短評' not in eval_df.columns: eval_df['短評'] = '-'
                            eval_df_clean = eval_df[['馬番_num', 'Gemini印', '短評']]
                            
                            res_df['馬番_num'] = pd.to_numeric(res_df['馬番'], errors='coerce')
                            
                            merged_df = pd.merge(res_df, eval_df_clean, on='馬番_num', how='left')
                            merged_df['Gemini印'] = merged_df['Gemini印'].fillna('消')
                            merged_df['短評'] = merged_df['短評'].fillna('-')
                            
                            table_placeholder.empty()
                            with table_placeholder.container():
                                st.markdown("<div class='section-header'>🔥 Python × Gemini 融合データ</div>", unsafe_allow_html=True)
                                st.markdown(generate_fusion_table(merged_df, is_newcomer), unsafe_allow_html=True)
                            
                            st.success("📝 調教・適性を加味した最強の印（骨格）をアップデートしました！")
                    except Exception as e:
                        st.error(f"❌ データの結合に失敗しました: {e}")
                else:
                    if not gemini_data:
                        st.warning("⚠️ 規定回数リトライしましたが、回答を正常に取得できませんでした。")