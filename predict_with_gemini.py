import os
import re
import time
import requests
import pandas as pd
import numpy as np
import joblib
import unicodedata
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
from google import genai
from google.genai import types

# ==========================================
# 🎨 アプリの基本設定
# ==========================================
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="🐴", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .section-header { font-size: 1.3rem; font-weight: bold; color: #2c3e50; margin: 1rem 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; white-space: nowrap; font-size: 14px; }
    .kachi-table th { padding: 8px; text-align: center; background: #fdfdfd; border-bottom: 2px solid #eaeaea; }
    .kachi-table td { padding: 8px; text-align: center; border-bottom: 1px solid #f4f4f4; }
    .badge-mark { color: #fff; padding: 4px 8px; border-radius: 4px; font-weight: bold; display: inline-block; min-width: 50px;}
    .badge-honmei { background: #e74c3c; } .badge-taikou { background: #3498db; }
    .badge-tana { background: #2ecc71; } .badge-renka { background: #f39c12; }
    .badge-ana { background: #9b59b6; } .badge-keshi { background: #e0e0e0; color: #7f8c8d; }
</style>
""", unsafe_allow_html=True)

st.title("🐴 AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 

FUTURE_CSV = "future_races.csv"
HISTORY_CSV = "prediction_history.csv"
ML_TARGET_CSV = "ml_target_data.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip()

# ==========================================
# 1. データとモデルの読み込み
# ==========================================
@st.cache_resource
def load_model():
    for m_name in ["勝ちパカくん.pkl", "keiba_ai_model.pkl"]:
        if os.path.exists(m_name):
            try: return joblib.load(m_name)
            except: pass
    return None

@st.cache_data
def load_data():
    # 過去データ
    df_past = pd.DataFrame()
    if os.path.exists(ML_TARGET_CSV):
        for enc in ['utf-8-sig', 'utf-8', 'cp932']:
            try:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding=enc)
                df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)
                df_past['date_parsed'] = pd.to_datetime(df_past['date'], errors='coerce')
                df_past = df_past.sort_values(by='date_parsed')
                break
            except: pass

    # 未来データ
    df_future = pd.DataFrame()
    if os.path.exists(FUTURE_CSV):
        for enc in ['utf-8-sig', 'utf-8', 'cp932']:
            try:
                df_f = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding=enc)
                if not df_f.empty:
                    df_f['race_id'] = df_f['race_id'].astype(str).str.zfill(12)
                    df_f['place_name'] = df_f['race_id'].str[4:6].map({"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}).fillna("開催場")
                    df_f['r_num'] = pd.to_numeric(df_f['race_id'].str[10:12], errors='coerce').fillna(1).astype(int)
                    df_f['馬名_clean'] = df_f['馬名'].astype(str).apply(clean_horse_name)
                    df_f['day_label'] = df_f['date'].astype(str).str.strip() if 'date' in df_f.columns else "当日"
                    df_future = df_f
                    break
            except: pass

    # 履歴データ
    df_hist = pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'result_pay'])
    if os.path.exists(HISTORY_CSV):
        try: df_hist = pd.read_csv(HISTORY_CSV, dtype=str, encoding='utf-8-sig')
        except: pass

    return df_past, df_future, df_hist

model_data = load_model()
df_past, df_future, df_history = load_data()

# ==========================================
# 📦 過去データ辞書化（直近の実績を引き出す）
# ==========================================
@st.cache_data
def build_past_horse_dict(df_p):
    if df_p.empty: return {}
    horse_dict = {}
    for horse, group in df_p.groupby('馬名_clean'):
        if not horse: continue
        last_row = group.iloc[-1]
        
        # 指数は直近3走の平均（欠損はNaN）
        t_vals = pd.to_numeric(group.get('my_time_idx', pd.Series()), errors='coerce').dropna()
        l_vals = pd.to_numeric(group.get('my_last3f_idx', pd.Series()), errors='coerce').dropna()
        p_vals = pd.to_numeric(group.get('my_pace_idx', pd.Series()), errors='coerce').dropna()
        s_vals = pd.to_numeric(group.get('my_start_idx', pd.Series()), errors='coerce').dropna()
        
        horse_dict[horse] = {
            'last_date': last_row['date_parsed'],
            'prev_prize': pd.to_numeric(last_row.get('賞金(万円)'), errors='coerce'),
            'prev_rank_num': pd.to_numeric(last_row.get('着順', last_row.get('prev_rank')), errors='coerce'),
            'eff_my_time_idx': t_vals.tail(3).mean() if not t_vals.empty else np.nan,
            'eff_my_last3f_idx': l_vals.tail(3).mean() if not l_vals.empty else np.nan,
            'eff_my_pace_idx': p_vals.tail(3).mean() if not p_vals.empty else np.nan,
            'eff_my_start_idx': s_vals.tail(3).mean() if not s_vals.empty else np.nan
        }
    return horse_dict

past_dict = build_past_horse_dict(df_past)

# ==========================================
# 3. AI推論ロジック（完全に1から再設計）
# ==========================================
def calculate_predictions(race_id_target, df_fut, cond):
    if df_fut.empty or model_data is None: return None
    race_df = df_fut[df_fut['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None

    model = model_data['model']
    features = model_data['features']

    # 1. 過去データの紐付け
    for col in ['last_date', 'prev_prize', 'prev_rank_num', 'eff_my_time_idx', 'eff_my_last3f_idx', 'eff_my_pace_idx', 'eff_my_start_idx']:
        race_df[col] = race_df['馬名_clean'].apply(lambda x: past_dict.get(x, {}).get(col, np.nan))

    # 2. 予測用特徴量の生成
    race_df['date_parsed_fut'] = pd.to_datetime(race_df['date'], errors='coerce')
    race_df['interval_days'] = (race_df['date_parsed_fut'] - race_df['last_date']).dt.days

    j_col = race_df.get('jockey_win_power', race_df.get('jockey_win_rate', pd.Series()))
    race_df['eff_jockey_win'] = pd.to_numeric(j_col, errors='coerce').clip(0.0, 1.0)
    race_df['eff_jockey_track_win'] = pd.to_numeric(race_df.get('jockey_track_win_rate'), errors='coerce').clip(0.0, 1.0)
    race_df['horse_win_rate_val'] = pd.to_numeric(race_df.get('horse_win_rate'), errors='coerce').clip(0.0, 1.0)
    race_df['horse_runs_val'] = pd.to_numeric(race_df.get('horse_runs'), errors='coerce')

    # 条件の設定
    race_df['condition_code'] = cond
    race_df['condition'] = cond

    # 3. 推論用データの準備（特徴量の順序と型を学習時と完全一致させる）
    X = pd.DataFrame(index=race_df.index)
    for f in features:
        if f in race_df.columns:
            X[f] = race_df[f]
        else:
            X[f] = np.nan

    # カテゴリ変数の処理
    cat_cols = ['surface_code', 'condition_code', 'sex_code']
    for cat in cat_cols:
        if cat in X.columns:
            # 簡易的なエンコーディング（学習時と同じ前提）
            X[cat] = X[cat].astype('category')
            if cat == 'sex_code': X[cat] = X[cat].replace({'牡':0,'牝':1,'セ':2})
            if cat == 'surface_code': X[cat] = X[cat].replace({'芝':0,'ダート':1,'障害':2})
            if cat == 'condition_code': X[cat] = X[cat].replace({'良':0,'稍重':1,'稍':1,'重':2,'不良':3})

    # 数値型に変換（NaNはそのまま）
    X_num = X.copy()
    for col in X_num.columns:
        if col not in cat_cols:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

    # 4. 予測の実行（生の確率を取得）
    try:
        raw_probs = model.predict(X_num)
    except Exception as e:
        st.error(f"モデル予測エラー: {e}")
        return None

    race_df['win_prob'] = raw_probs
    
    # 単勝オッズ
    race_df['単勝_num'] = pd.to_numeric(race_df.get('単勝', race_df.get('オッズ', pd.Series())), errors='coerce')
    race_df['人気'] = race_df['単勝_num'].rank(method='min')

    # 期待値
    race_df['ev'] = race_df['win_prob'] * race_df['単勝_num']

    # AIスコア（確率から偏差値を算出。平均50）
    mean_p = race_df['win_prob'].mean()
    std_p = race_df['win_prob'].std()
    if pd.isna(std_p) or std_p == 0:
        race_df['ai_score'] = 50
    else:
        race_df['ai_score'] = (50 + (race_df['win_prob'] - mean_p) / std_p * 10).round().astype(int)

    # 脚質の判定
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

    # 5. 印の決定（★完全実力主義：AIスコア順★）
    race_df = race_df.sort_values(by=['ai_score', 'ev'], ascending=[False, False]).reset_index(drop=True)
    race_df['印'] = "消"
    
    if len(race_df) > 0: race_df.loc[0, '印'] = "◎ 本命"
    if len(race_df) > 1: race_df.loc[1, '印'] = "◯ 対抗"
    if len(race_df) > 2: race_df.loc[2, '印'] = "▲ 単穴"
    if len(race_df) > 3: race_df.loc[3, '印'] = "△ 連下"
    if len(race_df) > 4: race_df.loc[4, '印'] = "△ 連下"
    
    # 穴馬探し（5番手以降でオッズ10倍以上で一番期待値が高い馬）
    if len(race_df) > 5:
        ana_candidates = race_df.iloc[5:][race_df['単勝_num'] >= 10.0]
        if not ana_candidates.empty:
            ana_idx = ana_candidates['ev'].idxmax()
            race_df.loc[ana_idx, '印'] = "☆ 穴馬"

    mark_order = {"◎ 本命":1, "◯ 対抗":2, "▲ 単穴":3, "△ 連下":4, "☆ 穴馬":5, "消":6}
    race_df['mark_rank'] = race_df['印'].map(mark_order).fillna(99)
    race_df = race_df.sort_values(by='mark_rank').reset_index(drop=True)

    return race_df

# ==========================================
# 4. メインUI
# ==========================================
st.sidebar.button("🔄 画面リロード", on_click=lambda: st.cache_data.clear(), use_container_width=True)

if df_future.empty:
    st.warning("出馬表が存在しません。")
    st.stop()

st.markdown("<div class='section-header'>🎯 レース選択</div>", unsafe_allow_html=True)
dates = sorted(df_future['day_label'].unique())
sel_date = st.radio("開催日", dates, horizontal=True, label_visibility="collapsed")
day_df = df_future[df_future['day_label'] == sel_date]
places = day_df['place_name'].unique()

tabs = st.tabs([f"📍 {p}" for p in places])
for i, place in enumerate(places):
    with tabs[i]:
        place_df = day_df[day_df['place_name'] == place]
        races = sorted(place_df['r_num'].unique())
        cols = st.columns(6)
        for j, r in enumerate(races):
            r_df = place_df[place_df['r_num'] == r]
            r_id = r_df['race_id'].iloc[0]
            r_name = str(r_df.get('race_name', [''])[0]).strip()
            label = f"{r}R {r_name}" if r_name and r_name != 'nan' else f"{r}R"
            if cols[j % 6].button(label, key=f"btn_{r_id}", use_container_width=True):
                st.session_state['selected_race_id'] = r_id

if st.session_state['selected_race_id']:
    t_id = str(st.session_state['selected_race_id'])
    t_rows = df_future[df_future['race_id'].astype(str) == t_id]
    if t_rows.empty: st.rerun()

    r_info = t_rows.iloc[0]
    r_name = r_info.get('race_name', '')
    st.markdown("---")
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{r_name}】</h2>", unsafe_allow_html=True)
    
    cond = st.radio("想定馬場", ["良", "稍重", "重", "不良"], horizontal=True)
    
    res_df = calculate_predictions(t_id, df_future, cond)
    
    if res_df is not None:
        st.markdown("<div class='section-header'>📊 予想データ</div>", unsafe_allow_html=True)
        
        # HTMLテーブル構築
        html = "<div class='table-container'><table class='kachi-table'>"
        html += "<tr><th>馬番</th><th>馬名</th><th>脚質</th><th>AIスコア</th><th>複勝率</th><th>オッズ</th><th>期待値</th><th>印</th></tr>"
        
        for _, r in res_df.iterrows():
            mark = r['印']
            b_cls = "badge-keshi"
            if "◎" in mark: b_cls = "badge-honmei"
            elif "◯" in mark: b_cls = "badge-taikou"
            elif "▲" in mark: b_cls = "badge-tana"
            elif "△" in mark: b_cls = "badge-renka"
            elif "☆" in mark: b_cls = "badge-ana"
            
            w_prob = r.get('win_prob', 0) * 100
            html += f"<tr>"
            html += f"<td><b>{int(r['馬番'])}</b></td>"
            html += f"<td><b>{r['馬名']}</b></td>"
            html += f"<td>{r['脚質']}</td>"
            html += f"<td><b>{r['ai_score']}</b></td>"
            html += f"<td>{w_prob:.1f}%</td>"
            html += f"<td>{r['単勝_num']}倍</td>"
            html += f"<td>{r['ev']:.2f}</td>"
            html += f"<td><span class='badge-mark {b_cls}'>{mark}</span></td>"
            html += f"</tr>"
        html += "</table></div>"
        
        st.markdown(html, unsafe_allow_html=True)