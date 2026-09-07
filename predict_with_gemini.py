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
    
    /* Gemini出力枠 */
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
# 🌟 EnsembleModelクラス
# ==========================================
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
        xgb_pred = self.xgb_model.predict(xgb.DMatrix(X_num))
        cat_pred = self.cat_model.predict(X_num)

        w1, w2, w3 = self.weights
        return w1 * lgb_pred + w2 * xgb_pred + w3 * cat_pred

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
    return None, f"モデルファイルが見つかりません。"

@st.cache_resource
def load_encoders():
    le_cond = joblib.load("le_cond.pkl") if os.path.exists("le_cond.pkl") else None
    le_surf = joblib.load("le_surf.pkl") if os.path.exists("le_surf.pkl") else None
    return le_cond, le_surf

@st.cache_data
def load_data():
    df_past = pd.DataFrame()
    past_error = None
    if os.path.exists(ML_TARGET_CSV):
        for enc in ['utf-8-sig', 'utf-8', 'cp932']:
            try:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding=enc)
                df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)
                df_past['date_parsed'] = pd.to_datetime(df_past['date'], errors='coerce')
                df_past['distance_num'] = pd.to_numeric(df_past.get('distance'), errors='coerce')
                df_past['dist_cat'] = df_past['distance_num'].apply(get_dist_cat)
                df_past['rank_num'] = pd.to_numeric(df_past.get('着順'), errors='coerce')
                df_past['is_win_past'] = (df_past['rank_num'] == 1).astype(int)
                
                place_code = df_past.get('place_code', pd.Series(['00']*len(df_past)))
                surface = df_past.get('surface', pd.Series(['芝']*len(df_past)))
                df_past['course_id'] = place_code.astype(str) + "_" + surface.astype(str) + "_" + df_past['distance_num'].fillna(0).astype(int).astype(str)
                df_past = df_past.sort_values(by='date_parsed')
                past_error = None
                break
            except Exception as e:
                past_error = f"'{ML_TARGET_CSV}' 読み込み失敗 ({enc}): {e}"

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

    return df_past, df_future, past_error, future_error

model_data, model_err = load_model()
le_cond, le_surf = load_encoders()
df_past, df_future, past_err, future_err = load_data()

# ==========================================
# 2. 過去データ辞書化 (🌟 EMA魔改造版)
# ==========================================
@st.cache_data
def build_past_horse_dict(df_p):
    if df_p.empty: return {}, {}, {}, {}
    horse_dict = {}
    
    df_p['騎手_clean'] = df_p['騎手'].astype(str).str.strip()
    jockey_map = df_p.dropna(subset=['rank_num']).groupby('騎手_clean')['is_win_past'].mean().to_dict()

    for horse, group in df_p.groupby('馬名_clean'):
        valid_past = group.dropna(subset=['rank_num'])
        if valid_past.empty: last_valid_row = group.iloc[-1]
        else: last_valid_row = valid_past.iloc[-1]
        
        def parse_pass_full(val):
            if pd.isna(val): return np.nan, np.nan, np.nan
            parts = str(val).split('-')
            try: return float(parts[0]), float(parts[-1]), float(parts[0]) - float(parts[-1])
            except: return np.nan, np.nan, np.nan
        
        p_1c, p_lc, p_cdiff = parse_pass_full(last_valid_row.get('通過', np.nan))
        
        # 🌟 魔改造: 賞金平均も直近5走を重視（EMA）
        prize_col = '賞金(万円)' if '賞金(万円)' in valid_past.columns else 'prize'
        prizes = pd.to_numeric(valid_past.get(prize_col, pd.Series()), errors='coerce').fillna(0)
        horse_prize_avg = prizes.ewm(span=5, min_periods=1).mean().iloc[-1] if not prizes.empty else 0.0
        
        # 🌟 魔改造: 条件別平均着順も直近3走を重視（EMA）
        cat_stats = {}
        for cat_name in ['sprint', 'mile_middle', 'stayer']:
            c_rows = valid_past[valid_past['dist_cat'] == cat_name]
            cat_stats[cat_name] = {'avg_rank': c_rows['rank_num'].ewm(span=3, min_periods=1).mean().iloc[-1] if len(c_rows) > 0 else 7.0}
            
        place_stats = {}
        for p_code in valid_past['place_code'].astype(str).unique():
            p_rows = valid_past[valid_past['place_code'].astype(str) == p_code]
            place_stats[p_code] = p_rows['rank_num'].ewm(span=3, min_periods=1).mean().iloc[-1] if len(p_rows) > 0 else 7.0 

        # 🌟 魔改造: 上がりの速さも直近3走を重視（EMA）
        l_vals = pd.to_numeric(valid_past.get('my_last3f_idx', pd.Series()), errors='coerce').dropna()
        last_3f_avg_rank = l_vals.ewm(span=3, min_periods=1).mean().iloc[-1] if not l_vals.empty else 50.0

        horse_dict[horse] = {
            'last_date': last_valid_row['date_parsed'],
            'last_kinryo': last_valid_row.get('kinryo_num', 55.0),
            'prev_dist': last_valid_row.get('distance_num', np.nan), 
            'horse_prize_avg': horse_prize_avg, 
            'prev_1c': p_1c if not pd.isna(p_1c) else 10.0, 
            'prev_last_corner': p_lc if not pd.isna(p_lc) else 10.0,
            'prev_corner_diff': p_cdiff if not pd.isna(p_cdiff) else 0.0,
            'cat_stats': cat_stats,
            'place_stats': place_stats, 
            'eff_my_start_idx': pd.to_numeric(valid_past.get('my_start_idx', pd.Series()), errors='coerce').tail(3).median(),
            'eff_my_last3f_idx': pd.to_numeric(valid_past.get('my_last3f_idx', pd.Series()), errors='coerce').tail(3).median(),
            'last_3f_avg_rank': last_3f_avg_rank
        }

    trainer_map = df_p.groupby('調教師')['is_win_past'].mean().to_dict()
    horse_track_map = df_p.groupby(['馬名_clean', 'place_code'])['is_win_past'].mean().to_dict()

    return horse_dict, trainer_map, horse_track_map, jockey_map

past_dict, trainer_map, horse_track_map, jockey_map = build_past_horse_dict(df_past)

# ==========================================
# 3. 発走時刻によるパドック判定関数
# ==========================================
def check_paddock_time(time_str):
    if not time_str or ':' not in str(time_str): return False, ""
    try:
        JST = timezone(timedelta(hours=+9), 'JST')
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
# 4. AI推論＆勝負気配算出ロジック
# ==========================================
def calculate_predictions(race_id_target, df_fut, cond):
    if df_fut.empty or model_data is None: return None, None, None, None
    race_df = df_fut[df_fut['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None, None, None, None

    model = model_data['model']
    features = model_data.get('features', [])

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

    target_cols = [
        'prev_dist', 'horse_prize_avg', 'prev_1c', 'last_3f_avg_rank', 
        'eff_my_start_idx', 'eff_my_last3f_idx', 'last_kinryo', 'prev_last_corner', 'prev_corner_diff'
    ]
    for col in target_cols:
        race_df[col] = race_df['馬名_clean'].apply(lambda x: past_dict.get(x, {}).get(col, np.nan))

    race_df['dist_change_num'] = race_df['distance_num'] - race_df['prev_dist'].fillna(race_df['distance_num'])
    race_df['same_dist_avg_rank'] = race_df.apply(lambda r: past_dict.get(r['馬名_clean'], {}).get('cat_stats', {}).get(r['dist_cat'], {}).get('avg_rank', 7.0), axis=1)
    race_df['same_place_avg_rank'] = race_df.apply(lambda r: past_dict.get(r['馬名_clean'], {}).get('place_stats', {}).get(r['place_code_str'], 7.0), axis=1)
    
    race_df['horse_prize_avg'] = race_df['horse_prize_avg'].fillna(0.0)
    race_df['race_avg_prize'] = race_df['horse_prize_avg'].mean()
    if race_df['race_avg_prize'].mean() == 0: race_df['race_avg_prize'] = 1.0
    race_df['race_prize_relative'] = race_df['horse_prize_avg'] / race_df['race_avg_prize']
    race_df['race_prize_rank'] = race_df['horse_prize_avg'].rank(ascending=False, method='min')

    race_df['first_corner'] = race_df['prev_1c']
    race_df['last_corner'] = race_df['prev_last_corner']
    race_df['corner_diff'] = race_df['prev_corner_diff']

    weights_parsed = race_df.get('馬体重', pd.Series([(np.nan, np.nan)]*len(race_df))).apply(parse_weight)
    race_df['body_weight'] = [p[0] for p in weights_parsed]
    race_df['kinryo_weight_ratio'] = race_df['kinryo_num'] / race_df['body_weight'].fillna(470)
        
    race_df['trainer_win_rate'] = race_df.get('調教師', pd.Series()).map(trainer_map).fillna(0.08)
    race_df['騎手_clean'] = race_df.get('騎手', pd.Series()).astype(str).str.strip()
    race_df['jockey_win_rate'] = race_df['騎手_clean'].map(jockey_map).fillna(0.1).clip(0.0, 1.0)
    race_df['horse_track_win_rate'] = race_df.apply(lambda r: horse_track_map.get((r.get('馬名_clean'), r.get('place_code_str')), 0.0), axis=1)
    
    race_df['interval_days'] = 30 

    # モデル推論
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
# 5. テーブル生成
# ==========================================
def generate_base_table(disp_df, is_newcomer):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>脚質</th><th>スコア</th><th>1着率</th><th>3着内率</th><th>オッズ</th><th>期待値</th><th>Python印</th></tr>"
    
    for _, r in disp_df.iterrows():
        ev_val, odds_val = float(r.get('ev', 0)), float(r.get('単勝_num', 0))
        win_val, top3_val = float(r.get('win_prob', 0)), float(r.get('top3_prob', 0))
        mark = r.get('印', '消')
        
        score_str = f"<b>{int(r.get('ai_score', 100))}</b>" if not is_newcomer else "-"
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        top3_str = f"<span style='color:#3498db; font-weight:bold;'>{top3_val*100:.1f}%</span>" if top3_val >= 0.50 else f"{top3_val*100:.1f}%"
        ev_str = f"<b>{ev_val:.2f}</b>" if not is_newcomer else "-"

        html += f"<tr>"
        html += f"<td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{r.get('脚質', '-')}</td><td>{score_str}</td><td>{win_str}</td><td>{top3_str}</td>"
        html += f"<td>{odds_val:.1f}倍</td><td>{ev_str}</td><td><span class='badge-mark {get_badge_class(mark)}'>{mark}</span></td>"
        html += "</tr>"
    html += "</table></div>"
    return html

def generate_fusion_table(merged_df, is_newcomer):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>AIｽｺア</th><th>1着率</th><th>期待値</th><th>Python印</th><th>Gemini印</th><th style='text-align:left;'>Gemini短評</th></tr>"
    
    for _, r in merged_df.iterrows():
        win_val = float(r.get('win_prob', 0))
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        
        html += f"<tr>"
        html += f"<td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td><b>{int(r.get('ai_score', 100))}</b></td><td>{win_str}</td><td><b>{float(r.get('ev', 0)):.2f}</b></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('印', '消'))}'>{r.get('印', '消')}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('Gemini印', '消'))}'>{r.get('Gemini印', '消')}</span></td>"
        html += f"<td style='text-align:left; font-size:15px; color:#444;'>{r.get('短評', '-')}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html

# ==========================================
# 6. メインUI
# ==========================================
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
                        
                        # 🌟 ここで改行文字を安全に埋め込む
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

            table_summary = [f"馬番:{int(r.get('馬番',0)):02d} | 馬名:{r.get('馬名','')} | 脚質:{r.get('脚質','')}" for _, r in res_df.iterrows()]

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
1. 提供された「出走馬リスト」を確認してください（過去のオッズ・勝率は見ないこと）。
2. Google検索ツールを駆使して、以下の【定性情報】を検索してください。
   ①【陣営の勝負気配】: メイチか叩き台か。
   ②【馬場適性】: 今日のコース・馬場状態に対する裏付け。
   ③【前走不利】: 前走大敗だが今回巻き返せる「隠れた実力馬」の発掘。
{paddock_instruction}
3. 検索で得た定性情報のみを基に、最終評価を下してください。

【印の打ち方】
◎(1頭), ◯(1頭), ▲(1頭), △(1頭), ☆(1〜2頭), 消(それ以外)
※テキスト解説不要。必ず以下のJSONのみ出力すること。
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