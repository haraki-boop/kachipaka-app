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
    
    .sense-card { background-color: #ffffff; border-left: 6px solid #8e44ad; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .win5-card { background-color: #fff9e6; border-left: 6px solid #f1c40f; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
    .gemini-win5-box { background-color: #f8f9fa; border: 2px solid #f1c40f; border-radius: 8px; padding: 20px; margin-top: 10px; line-height: 1.6; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("🐴 AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_FILE = "keiba_ai_model.pkl"
FUTURE_CSV = "future_races.csv"
CACHE_FILE = "app_cache.pkl"

# ==========================================
# 📊 馬場・コースバイアス補正設定
# ==========================================
TRACK_BIAS = {
    "重・不良": {"逃げ": 1.20, "先行": 1.10, "差し": 0.95, "追込": 0.85},
    "稍重": {"逃げ": 1.10, "先行": 1.05, "差し": 1.00, "追込": 0.95},
    "中山": {"逃げ": 1.15, "先行": 1.05, "差し": 0.90, "追込": 0.80}, 
    "函館": {"逃げ": 1.15, "先行": 1.10, "差し": 0.85, "追込": 0.80}, 
    "札幌": {"逃げ": 1.15, "先行": 1.10, "差し": 0.85, "追込": 0.80}, 
    "東京": {"逃げ": 0.90, "先行": 0.95, "差し": 1.15, "追込": 1.10}, 
    "新潟": {"逃げ": 0.90, "先行": 0.95, "差し": 1.15, "追込": 1.10}, 
    "阪神": {"逃げ": 0.95, "先行": 1.00, "差し": 1.10, "追込": 1.00}, 
    "京都": {"逃げ": 1.05, "先行": 1.05, "差し": 1.00, "追込": 0.90}, 
    "小倉": {"逃げ": 1.10, "先行": 1.05, "差し": 0.90, "追込": 0.85}, 
    "福島": {"逃げ": 1.15, "先行": 1.05, "差し": 0.90, "追込": 0.80}, 
    "中京": {"逃げ": 1.00, "先行": 1.05, "差し": 1.05, "追込": 0.95}, 
}

def clean_name(text):
    if pd.isna(text): return ""
    s = unicodedata.normalize('NFKC', str(text))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip().upper()

def get_badge_class(mark):
    if pd.isna(mark): return "badge-keshi"
    if "◎" in mark: return "badge-honmei"
    elif "◯" in mark: return "badge-taikou"
    elif "▲" in mark: return "badge-tana"
    elif "△" in mark: return "badge-renka"
    elif "☆" in mark: return "badge-ana"
    return "badge-keshi"

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

        return self.weights[0] * lgb_pred + self.weights[1] * xgb_pred + self.weights[2] * cat_pred

import __main__
__main__.EnsembleModel = EnsembleModel

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE), None
    return None, "モデルファイルが見つかりません。"

@st.cache_data
def load_future_data():
    if os.path.exists(FUTURE_CSV):
        df_f = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
        df_f['race_id'] = df_f['race_id'].astype(str).str.zfill(12)
        df_f['place_name'] = df_f['race_id'].str[4:6].map({"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}).fillna("開催場")
        df_f['r_num'] = pd.to_numeric(df_f['race_id'].str[-2:], errors='coerce').fillna(1).astype(int)
        df_f['day_label'] = df_f['date'].astype(str).str.strip() if 'date' in df_f.columns else "当日"
        df_f['馬名_clean'] = df_f['馬名'].astype(str).apply(clean_name)
        df_f['騎手_clean'] = df_f['騎手'].astype(str).apply(clean_name)
        df_f['調教師_clean'] = df_f['調教師'].astype(str).str.replace(r'\[.*?\]\n', '', regex=True).apply(clean_name)
        return df_f, None
    return pd.DataFrame(), "出馬表データが見つかりません。"

@st.cache_resource
def load_cache():
    if os.path.exists(CACHE_FILE):
        return joblib.load(CACHE_FILE), None
    return None, "キャッシュが見つかりません。"

model_data, model_err = load_model()
df_future, future_err = load_future_data()
app_cache, cache_err = load_cache()

# ==========================================
# 🧠 推論ロジック (最強結合＆動的特徴量生成)
# ==========================================
def calculate_predictions(race_id_target, cond):
    if df_future.empty or model_data is None or app_cache is None: return None, None, None, None
    df = df_future[df_future['race_id'].astype(str) == str(race_id_target)].copy()
    if df.empty: return None, None, None, None

    model = model_data['model']
    features = model_data['features']

    # 🚨 全キャッシュ辞書の強制マージ
    for key, value in app_cache.items():
        if isinstance(value, dict) and len(value) > 0:
            first_val = next(iter(value.values()))
            if isinstance(first_val, dict):
                temp_df = pd.DataFrame.from_dict(value, orient='index').reset_index()
                temp_df = temp_df.rename(columns={'index': '馬名_clean'})
                cols_to_use = temp_df.columns.difference(df.columns).tolist() + ['馬名_clean']
                df = pd.merge(df, temp_df[cols_to_use], on='馬名_clean', how='left')

    df['騎手_win_rate'] = df['騎手_clean'].map(app_cache.get('jockey_win_map', {})).fillna(0.05)
    df['trainer_win_rate'] = df['調教師_clean'].map(app_cache.get('trainer_win_map', {})).fillna(0.05)

    # 🚨 動的特徴量の計算
    if 'last_date' in df.columns:
        last_dates = pd.to_datetime(df['last_date'], errors='coerce', utc=True).dt.tz_convert(None)
        curr_dates = pd.Series([pd.Timestamp.now()] * len(df), index=df.index)
        df['interval_days'] = (curr_dates - last_dates).dt.days.fillna(30)
    else:
        df['interval_days'] = 30

    if 'prev_1c' in df.columns:
        df['prev_1c'] = pd.to_numeric(df['prev_1c'], errors='coerce').fillna(10.0)
        df['race_expected_pace'] = df.groupby('race_id')['prev_1c'].transform('mean').fillna(10.0)
        df['pace_advantage'] = df['prev_1c'] - df['race_expected_pace']
    else:
        df['race_expected_pace'] = 10.0
        df['pace_advantage'] = 0.0

    if '斤量' in df.columns:
        df['kinryo_num'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(55.0)
    else:
        df['kinryo_num'] = 55.0

    for idx_type in ['pace_idx', 'last3f_idx', 'start_idx']:
        cols = [f'prev{i}_{idx_type}' for i in range(1, 6)]
        valid_cols = [c for c in cols if c in df.columns]
        if valid_cols:
            df[f'max_{idx_type}'] = df[valid_cols].astype(float).max(axis=1).fillna(0.0)
            if f'prev1_{idx_type}' in df.columns:
                df[f'ratio_to_max_{idx_type}'] = pd.to_numeric(df[f'prev1_{idx_type}'], errors='coerce') / df[f'max_{idx_type}'].replace(0, 1.0)
                df[f'ratio_to_max_{idx_type}'] = df[f'ratio_to_max_{idx_type}'].fillna(0.0)
        else:
            df[f'max_{idx_type}'] = 0.0
            df[f'ratio_to_max_{idx_type}'] = 0.0

    base_cols = []
    for f in features:
        if f.endswith('_race_diff'): base_cols.append(f.replace('_race_diff', ''))
        elif f.endswith('_race_zscore'): base_cols.append(f.replace('_race_zscore', ''))
        elif f.endswith('_race_rank'): base_cols.append(f.replace('_race_rank', ''))
        elif f.endswith('_race_ratio'): base_cols.append(f.replace('_race_ratio', ''))
    
    base_cols = list(set(base_cols))
    for b in base_cols:
        if b not in df.columns: df[b] = 0.0
        else: df[b] = pd.to_numeric(df[b], errors='coerce').fillna(0.0)
        mean_val = df.groupby('race_id')[b].transform('mean')
        std_val = df.groupby('race_id')[b].transform('std').replace(0, 1.0)
        df[f'{b}_race_diff'] = df[b] - mean_val
        df[f'{b}_race_zscore'] = (df[b] - mean_val) / std_val
        df[f'{b}_race_rank'] = df.groupby('race_id')[b].rank(ascending=False, method='min')
        df[f'{b}_race_ratio'] = df[b] / mean_val.replace(0, 1.0)

    X_future = pd.DataFrame(index=df.index)
    for f in features:
        X_future[f] = pd.to_numeric(df[f], errors='coerce') if f in df.columns else 0.0

    df['raw_score'] = model.predict(X_future)

    # 勝率の算出
    raw_scores = df['raw_score'].values
    s_std = np.std(raw_scores)
    if s_std > 0:
        z_scores = (raw_scores - np.mean(raw_scores)) / s_std
        base_probs = 1.0 / (1.0 + np.exp(-1.2 * z_scores))
        df['win_prob'] = base_probs * 0.35 + 0.01
    else:
        df['win_prob'] = 0.10

    # 🐎 脚質の判定 (修正: データが存在する場合のみ正確にランク付け)
    total_horses = len(df)
    if total_horses > 0 and 'ema3_start_idx' in df.columns:
        df['ema3_start_idx'] = pd.to_numeric(df['ema3_start_idx'], errors='coerce')
        df['start_rank'] = df['ema3_start_idx'].rank(ascending=False, method='min', na_option='bottom')
        def det_style(row):
            if pd.isna(row.get('ema3_start_idx')) or row.get('ema3_start_idx') == 0:
                return "-"
            pct = row['start_rank'] / total_horses
            if pct <= 0.20: return "逃げ"
            elif pct <= 0.45: return "先行"
            elif pct <= 0.75: return "差し"
            else: return "追込"
        df['脚質'] = df.apply(det_style, axis=1)
    else:
        df['脚質'] = "-"

    # 馬場・コースバイアス補正
    df['bias_coef'] = 1.0
    place_name = str(df['place_name'].iloc[0])

    if cond in ['重', '不良']:
        df['bias_coef'] *= df['脚質'].map(TRACK_BIAS["重・不良"]).fillna(1.0)
    elif cond == '稍重':
        df['bias_coef'] *= df['脚質'].map(TRACK_BIAS["稍重"]).fillna(1.0)
    
    if place_name in TRACK_BIAS:
        df['bias_coef'] *= df['脚質'].map(TRACK_BIAS[place_name]).fillna(1.0)

    df['win_prob'] = df['win_prob'] * df['bias_coef']
    sum_prob = df['win_prob'].sum()
    if sum_prob > 0: df['win_prob'] = df['win_prob'] / sum_prob

    min_p, max_p = df['win_prob'].min(), df['win_prob'].max()
    if max_p > min_p:
        df['ai_score'] = (50 + (df['win_prob'] - min_p) / (max_p - min_p) * 100).round().astype(int)
    else:
        df['ai_score'] = 100

    df = df.sort_values(by=['win_prob'], ascending=False).reset_index(drop=True)
    
    marks = ["◎", "◯", "▲", "△", "☆1", "☆2"]
    df['印'] = "消"
    for i in range(min(len(df), len(marks))): df.loc[i, '印'] = marks[i]

    probs = df['win_prob'].values
    p1 = probs[0] if len(probs)>0 else 0
    p2 = probs[1] if len(probs)>1 else 0
    p3 = probs[2] if len(probs)>2 else 0
    p4 = probs[3] if len(probs)>3 else 0.05
    
    gap_1_2 = p1 - p2
    gap_1_3 = p1 - p3

    if gap_1_2 >= 0.07:
        pat = "① 1強気配 (軸圧倒)"
        rec_ticket = "3連単 1着固定 (6点)"
        buy_detail = f"1着: {df.loc[0, '馬番']}(◎) -> 2・3着: {df.loc[1, '馬番']}(◯), {df.loc[2, '馬番']}(▲), {df.loc[3, '馬番']}(△)"
    elif gap_1_2 < 0.035 and gap_1_3 >= 0.06:
        pat = "② 2強気配 (頭分け対抗)"
        rec_ticket = "3連単 ダブル軸 (12点)"
        buy_detail = f"1着: ◎, ◯ -> 2着: ◎, ◯, ▲ -> 3着: ◎, ◯, ▲, △"
    elif (p1 - p4) < 0.08:
        pat = "④ 波乱気配 (大混戦)"
        rec_ticket = "3連複 5頭BOX (10点)"
        buy_detail = "BOX: " + ", ".join([str(df.loc[k, '馬番']) for k in range(min(5, len(df)))])
    else:
        pat = "③ 混戦気配 (標準展開)"
        rec_ticket = "3連複 ◎1頭軸流し (10点)"
        buy_detail = f"軸: {df.loc[0, '馬番']}(◎) -> 相手: " + ", ".join([str(df.loc[k, '馬番']) for k in range(1, min(6, len(df)))])

    return df, pat, rec_ticket, buy_detail

# ==========================================
# 📊 UI表示部
# ==========================================
def generate_base_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>脚質</th><th>斤量</th><th>スコア</th><th>1着率</th><th>補正係数</th><th>Python印</th></tr>"
    for _, r in disp_df.iterrows():
        win_val = float(r.get('win_prob', 0))
        kinryo_val = float(r.get('斤量', 55.0))
        coef_val = float(r.get('bias_coef', 1.0))
        score_str = f"<b>{int(r.get('ai_score', 100))}</b>"
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        
        coef_color = "color:#e74c3c; font-weight:bold;" if coef_val > 1.0 else "color:#3498db;" if coef_val < 1.0 else "color:#95a5a6;"
        coef_str = f"<span style='{coef_color}'>x{coef_val:.2f}</span>"
        
        mark = r.get('印', '消')
        html += f"<tr><td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{r.get('脚質', '-')}</td><td>{kinryo_val:.1f}kg</td><td>{score_str}</td><td>{win_str}</td><td>{coef_str}</td>"
        html += f"<td><span class='badge-mark {get_badge_class(mark)}'>{mark}</span></td></tr>"
    html += "</table></div>"
    return html

def generate_fusion_table(merged_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>AIｽｺア</th><th>1着率</th><th>Python印</th><th>Gemini印</th><th style='text-align:left;'>Gemini短評</th></tr>"
    
    for _, r in merged_df.iterrows():
        win_val = float(r.get('win_prob', 0))
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        
        html += f"<tr>"
        html += f"<td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td><b>{int(r.get('ai_score', 100))}</b></td><td>{win_str}</td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('印', '消'))}'>{r.get('印', '消')}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('Gemini印', '消'))}'>{r.get('Gemini印', '消')}</span></td>"
        html += f"<td style='text-align:left; font-size:15px; color:#444;'>{r.get('短評', '-')}</td>"
        html += "</tr>"
    html += "</table></div>"
    return html

st.sidebar.button("🔄 画面リロード", on_click=lambda: st.cache_data.clear(), use_container_width=True)
st.markdown("<div class='section-header'>🎯 レース選択</div>", unsafe_allow_html=True)

if df_future.empty:
    st.warning("出馬表データがありません。scrape_shutsuba.py を実行してください。")
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
                    
                    r_df, _, _, _ = calculate_predictions(r_id, "良")
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
                まずは必ず「JRA {sel_date} WIN5 対象レース」でGoogle検索を行い、本日の【公式のWIN5対象の5レース】を正確に特定してください。
                
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
                    st.markdown(f"<div class='gemini-win5-box'>{response.text}</div>", unsafe_allow_html=True)
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
    st.markdown("---")
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{r_info.get('race_name', '')}】</h2>", unsafe_allow_html=True)
    
    cond = st.radio("想定馬場 (AIのコースバイアス補正に反映されます)", ["良", "稍重", "重", "不良"], horizontal=True)
    
    res_df, pat, rec_ticket, buy_detail = calculate_predictions(t_id, cond)
    
    if res_df is not None:
        st.markdown(f"""
        <div class='sense-card'>
            <div class='sense-title'>🧠 AI勝負気配判定: {pat}</div>
            <div style='margin-top: 8px;'><b>🎟️ 推奨買い目:</b> <span class='ticket-badge'>{rec_ticket}</span></div>
            <div style='margin-top: 6px; font-size: 15px; color: #555;'><b>📝 買い目詳細:</b> {buy_detail}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Pythonベース予想データ (馬場・コース補正済み)</div>", unsafe_allow_html=True)
        st.info("※ 「補正係数」は、選択した馬場状態と競馬場のコース形態から算出された脚質有利・不利の度合いです（1.0が基準）。")
        st.markdown(generate_base_table(res_df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🧠 Geminiを独立させて、泥臭く穴馬を発掘させる", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            table_summary = []
            for _, r in res_df.iterrows():
                info = f"馬番:{int(r.get('馬番',0)):02d} | 馬名:{r.get('馬名','')} | 脚質:{r.get('脚質','')}"
                table_summary.append(info)

            system_instruction = f"""
あなたはプロ競馬予想家（トラックマン）です。
【あなたのワークフロー】
1. 提供された「出走馬リスト」を確認してください。
2. Google検索ツールを駆使して、以下の【定性情報】を最優先で検索・収集してください。
   ①【調教・勝負気配】: 今回はメイチ（本気）か、次を見据えた叩き台か。最終追い切りの動き。
   ②【血統・馬場適性】: 今日のコースや馬場状態に対する血統的な裏付け。
   ③【直前気配】: X(Twitter)等での現地からの直前パドック情報や馬体増減のニュアンス。
3. 検索で得た定性情報のみを基に、最終評価を下してください。

【印の打ち方】
◎(1頭), ◯(1頭), ▲(1頭), △(1頭), ☆(1〜2頭), 消(それ以外)
※必ず以下のJSONのみ出力すること。
{{ "evaluations": [ {{"馬番": 1, "Gemini印": "◎", "短評": "〇〇のため好走必至"}}, ... ] }}
"""
            prompt = f"対象レース: {sel_date} {r_info['place_name']} {r_info['r_num']}R\n【想定馬場】: {cond}\n【出走馬リスト】:\n{chr(10).join(table_summary)}"

            with st.spinner("🧠 GeminiがPythonに頼らず、独自の視点で検索中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                gemini_data = None
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction, temperature=0.7, tools=[{"googleSearch": {}}]
                        )
                    )
                    res_text = response.text if response.text else ""
                    match = re.search(r'\{.*\}', res_text, re.DOTALL)
                    gemini_data = json.loads(match.group(0) if match else res_text)
                except Exception as e:
                    st.error(f"❌ APIエラー: {e}")
                    
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
                        
                        st.markdown("<div class='section-header'>🔥 Python（確率脳） × Gemini（定性脳） 独立評価比較</div>", unsafe_allow_html=True)
                        st.markdown(generate_fusion_table(merged_df), unsafe_allow_html=True)