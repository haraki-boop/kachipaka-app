import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import unicodedata
import streamlit as st
from google import genai
from google.genai import types

# ==========================================
# 🎨 アプリの基本設定 & スタイル定義
# ==========================================
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="logo.png", layout="wide")

st.markdown("""
<style>
    /* 🎯 背景グラデーション */
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #ffffff 100%);
    }
    
    .block-container { padding-top: 3.5rem; padding-bottom: 2rem; }
    
    .section-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; margin: 1.2rem 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; }
    .kachi-table { width: 100%; border-collapse: collapse; background-color: #ffffff; white-space: nowrap; font-size: 17px; }
    .kachi-table th { padding: 12px 10px; text-align: center; background: #f8f9fa; border-bottom: 2px solid #dee2e6; font-size: 17px; font-weight: bold; }
    .kachi-table td { padding: 12px 10px; text-align: center; border-bottom: 1px solid #f1f3f5; font-size: 17px; }
    .badge-mark { color: #fff; padding: 6px 12px; border-radius: 6px; font-weight: bold; display: inline-block; min-width: 60px; font-size: 16px; }
    .badge-honmei { background: #e74c3c; } .badge-taikou { background: #3498db; }
    .badge-tana { background: #2ecc71; } .badge-renka { background: #f39c12; }
    .badge-ana { background: #9b59b6; } .badge-keshi { background: #e0e0e0; color: #7f8c8d; }
    
    .sense-card { background-color: #ffffff; border-left: 6px solid #8e44ad; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
    
    .gemini-view-card { background-color: #f0f8ff; border-left: 5px solid #3498db; padding: 16px 20px; border-radius: 5px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .gemini-view-title { color: #2980b9; font-size: 1.2rem; font-weight: bold; margin-bottom: 8px; }
    .gemini-view-text { font-size: 16px; color: #333; line-height: 1.6; margin: 0; }
    
    /* 🚨 カスタムボタンデザイン（青系・グラデーション） */
    div.stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #1f618d 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FUTURE_CSV = "future_races.csv"
CACHE_FILE = "app_cache.pkl"
BIAS_CSV = "course_bias.csv"
WAKU_CSV = "all_races_waku_stats.csv"

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

def normalize_kyakushitsu(k_str):
    k = str(k_str).strip()
    if pd.isna(k_str) or k in ['nan', 'NaN', 'None', '', '-']:
        return "不明"
    if "大" in k or "逃" in k: return "逃げ"
    if "先" in k: return "先行"
    if "差" in k: return "差し"
    if "追" in k: return "追込"
    if "マ" in k: return "自在"
    return "不明"

@st.cache_data
def load_future_data():
    if os.path.exists(FUTURE_CSV):
        df_f = pd.read_csv(FUTURE_CSV, dtype={'race_id': str, '枠番': str, '馬番': str}, encoding='utf-8-sig')
        df_f['race_id'] = df_f['race_id'].astype(str).str.zfill(12)
        df_f['place_name'] = df_f['race_id'].str[4:6].map({"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}).fillna("開催場")
        df_f['r_num'] = pd.to_numeric(df_f['race_id'].str[-2:], errors='coerce').fillna(1).astype(int)
        df_f['day_label'] = df_f['date'].astype(str).str.strip() if 'date' in df_f.columns else "当日"
        df_f['馬名_clean'] = df_f['馬名'].astype(str).apply(clean_name)
        
        if '推測脚質' in df_f.columns:
            df_f['推測脚質'] = df_f['推測脚質'].apply(normalize_kyakushitsu)
        else:
            df_f['推測脚質'] = "不明"

        for i, row in df_f.iterrows():
            if pd.isna(row['馬番']) or str(row['馬番']).strip() in ["", "nan", "NaN"]:
                df_f.at[i, '馬番'] = str(90 + i)
        
        return df_f
    return pd.DataFrame()

@st.cache_data
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return joblib.load(CACHE_FILE)
        except Exception:
            return {}
    return {}

def load_course_bias():
    if os.path.exists(BIAS_CSV):
        return pd.read_csv(BIAS_CSV, dtype={'距離': str}, encoding='utf-8-sig')
    else:
        df = pd.DataFrame(columns=['競馬場', '芝ダート', '距離', '内枠', '外枠', '逃げ', '先行', '差し', '追込'])
        df.loc[0] = ['東京', '芝', '1600', 1.05, 0.95, 0.90, 0.95, 1.10, 1.05]
        df.to_csv(BIAS_CSV, index=False, encoding='utf-8-sig')
        return df

def load_waku_stats():
    if os.path.exists(WAKU_CSV):
        return pd.read_csv(WAKU_CSV, dtype={'距離': str}, encoding='utf-8-sig')
    return pd.DataFrame()

df_future = load_future_data()
df_course_bias = load_course_bias()

def calculate_mixed_bias(row, place_name, surface, distance, cond, race_day, df_bias):
    coef = 1.0
    kyaku = str(row.get('推測脚質', '不明')).strip()
    waku_str = str(row.get('枠番', ''))
    
    waku_num = int(waku_str) if waku_str.isdigit() else 0
    is_uchi = 1 <= waku_num <= 3
    is_soto = 6 <= waku_num <= 8
    
    dist_str = re.sub(r'\D', '', str(distance))
    if not dist_str: dist_str = "1600"
    
    search_surface = 'ダート' if 'ダ' in str(surface) else '芝'

    if not df_bias.empty:
        match = df_bias[(df_bias['競馬場'] == place_name) & 
                        (df_bias['芝ダート'] == search_surface) & 
                        (df_bias['距離'] == dist_str)]
        
        if not match.empty:
            b = match.iloc[0]
            if is_uchi and pd.notna(b.get('内枠')): coef *= float(b['内枠'])
            if is_soto and pd.notna(b.get('外枠')): coef *= float(b['外枠'])
            
            if kyaku == '逃げ' and pd.notna(b.get('逃げ')): coef *= float(b['逃げ'])
            elif kyaku == '先行' and pd.notna(b.get('先行')): coef *= float(b['先行'])
            elif kyaku == '差し' and pd.notna(b.get('差し')): coef *= float(b['差し'])
            elif kyaku == '追込' and pd.notna(b.get('追込')): coef *= float(b['追込'])

    if search_surface == '芝':
        if cond in ['重', '不良']:
            if kyaku in ['逃げ', '先行']: coef *= 1.05
            elif kyaku in ['差し', '追込']: coef *= 0.95
    elif search_surface == 'ダート':
        if cond in ['稍重', '重', '不良']:
            if kyaku in ['逃げ', '先行']: coef *= 1.08 
            elif kyaku in ['追込']: coef *= 0.90

    if waku_num > 0 and search_surface == '芝':
        if race_day <= 4:
            if is_uchi: coef *= 1.05
        elif race_day >= 9:
            if is_soto: coef *= 1.05

    return round(max(0.70, min(coef, 1.50)), 2)

def render_race_predictions(race_id_target, cond):
    if df_future.empty: return None, None, None, None, None
    
    cache_data = load_cache()
    if not cache_data:
        st.warning("⚠️ キャッシュファイル(app_cache.pkl)が見つかりません。")
        return None, None, None, None, None
    
    df = df_future[df_future['race_id'].astype(str) == str(race_id_target)].copy()
    if df.empty: return None, None, None, None, None

    all_preds = cache_data.get('predictions', {})
    target_key = str(race_id_target).zfill(12)
    predictions = all_preds.get(target_key) or all_preds.get(str(race_id_target))
    
    if not predictions:
        st.error(f"⚠️ レースID '{target_key}' の計算済みデータが見つかりません。推論を実行してください。")
        return None, None, None, None, None

    horse_preds = predictions.get('horses', {})
    
    df['win_prob'] = df['馬名_clean'].map(lambda x: horse_preds.get(x, {}).get('win_prob', 0.1))
    
    place_name = str(df['place_name'].iloc[0])
    surface = str(df['surface'].iloc[0])
    distance = str(df['distance'].iloc[0])
    
    try:
        race_day = int(target_key[8:10])
    except:
        race_day = 1

    df['bias_coef'] = df.apply(lambda row: calculate_mixed_bias(row, place_name, surface, distance, cond, race_day, df_course_bias), axis=1)

    df['win_prob'] = df['win_prob'] * df['bias_coef']
    sum_p = df['win_prob'].sum()
    if sum_p > 0: df['win_prob'] = df['win_prob'] / sum_p

    min_p, max_p = df['win_prob'].min(), df['win_prob'].max()
    if max_p > min_p:
        df['ai_score'] = (50 + (df['win_prob'] - min_p) / (max_p - min_p) * 100).round().astype(int)

    df = df.sort_values(by=['win_prob'], ascending=False).reset_index(drop=True)

    mark_list = ['◎', '◯', '▲', '△', '☆', '☆']
    new_marks = []
    for i in range(len(df)):
        new_marks.append(mark_list[i] if i < len(mark_list) else '消')
    df['印'] = new_marks

    pat = predictions.get('pattern', '③ 混戦気配')
    rec_ticket = predictions.get('ticket', '3連複')
    
    if len(df) >= 6:
        h1 = df.iloc[0]
        def fmt_m(val):
            v_str = str(val)
            return "未定" if v_str.startswith("90") or int(float(v_str)) >= 90 else f"{int(float(v_str)):02d}"
        
        aite_str = ", ".join([f"{fmt_m(df.iloc[i]['馬番'])}({df.iloc[i]['馬名']})({df.iloc[i]['印']})" for i in range(1, 6)])
        buy_detail = f"軸: {fmt_m(h1['馬番'])}({h1['馬名']})(◎) -> 相手: {aite_str}"
    else:
        buy_detail = predictions.get('buy_detail', '')

    return df, pat, rec_ticket, buy_detail, race_day

def generate_base_table(disp_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>枠番</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>推測脚質</th><th>斤量</th><th>騎手</th><th>スコア</th><th>1着率</th><th>補正係数</th><th>Python印</th></tr>"
    for _, r in disp_df.iterrows():
        w_num = str(r.get('枠番', ''))
        display_waku = w_num if w_num and w_num != 'nan' and not w_num.startswith("90") else "-"
        
        m_num = str(r['馬番'])
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        
        win_val = float(r.get('win_prob', 0))
        kinryo_val = float(r.get('斤量', 55.0))
        coef_val = float(r.get('bias_coef', 1.0))
        score_str = f"<b>{int(r.get('ai_score', 100))}</b>"
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        
        coef_color = "color:#e74c3c; font-weight:bold;" if coef_val > 1.0 else "color:#3498db;" if coef_val < 1.0 else "color:#95a5a6;"
        coef_str = f"<span style='{coef_color}'>x{coef_val:.2f}</span>"
        
        mark = r.get('印', '消')
        
        style_raw = str(r.get('推測脚質', '不明')).strip()
        if style_raw == '不明':
            style_html = "<span style='color:#bdc3c7;'>-</span>"
        elif style_raw == '逃げ':
            style_html = "<span style='color:#e74c3c; font-weight:bold;'>逃げ</span>"
        else:
            style_html = style_raw
            
        jockey_val = str(r.get('騎手', '-')).strip()
        
        html += f"<tr><td><b>{display_waku}</b></td><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{style_html}</td><td>{kinryo_val:.1f}kg</td><td>{jockey_val}</td><td>{score_str}</td><td>{win_str}</td><td>{coef_str}</td>"
        html += f"<td><span class='badge-mark {get_badge_class(mark)}'>{mark}</span></td></tr>"
    html += "</table></div>"
    return html

def generate_fusion_table(merged_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>枠番</th><th>馬番</th><th style='text-align:left;'>馬名</th><th>AIｽｺア</th><th>調教評価</th><th>Python印</th><th>Gemini印</th><th style='text-align:left; min-width: 400px;'>Gemini短評 (展開・定性評価)</th></tr>"
    for _, r in merged_df.iterrows():
        w_num = str(r.get('枠番', ''))
        display_waku = w_num if w_num and w_num != 'nan' and not w_num.startswith("90") else "-"
        
        m_num = str(r.get('馬番', ''))
        display_umaban = "未定" if m_num.startswith("90") or int(float(m_num)) >= 90 else f"{int(float(m_num)):02d}"
        
        pros_cons = r.get('pros_cons', '')
        tanpyo = r.get('短評', '-')
        
        if pd.notna(pros_cons) and str(pros_cons).strip() != '':
            combined_text = f"<div style='color:#7f8c8d; font-size:13px; margin-bottom:4px;'>{pros_cons}</div><div style='font-weight:bold; color:#2c3e50;'>📝 {tanpyo}</div>"
        else:
            combined_text = tanpyo
            
        chokyo_eval = str(r.get('調教評価', '-'))
            
        html += f"<tr><td><b>{display_waku}</b></td><td><b>{display_umaban}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td><b>{int(r.get('ai_score', 100))}</b></td>"
        html += f"<td><span style='font-weight:bold; color:#d35400;'>{chokyo_eval}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('印', '消'))}'>{r.get('印', '消')}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('Gemini印', '消'))}'>{r.get('Gemini印', '消')}</span></td>"
        html += f"<td style='text-align:left; font-size:14px; white-space:normal; line-height:1.5; padding: 12px;'>{combined_text}</td></tr>"
    html += "</table></div>"
    return html

def clear_all_caches():
    st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()

# ==========================================
# 🎯 画面表示
# ==========================================
st.sidebar.button("🔄 画面リロード", on_click=clear_all_caches, use_container_width=True)

col_logo, col_main = st.columns([1, 4])

with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

with col_main:
    st.markdown("<div class='section-header' style='margin-top: 0;'>🎯 レース選択</div>", unsafe_allow_html=True)

    if df_future.empty:
        st.warning("出馬表データがありません。scrape_shutsuba.py を実行してください。")
        st.stop()

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
                        if cols[k].button(f"{r}R {r_name}" if r_name and r_name!='nan' else f"{r}R", key=f"btn_{r_id}", use_container_width=True):
                            st.session_state['selected_race_id'] = r_id

if st.session_state['selected_race_id']:
    t_id = str(st.session_state['selected_race_id'])
    t_rows = df_future[df_future['race_id'].astype(str) == t_id]
    if t_rows.empty: st.stop()

    r_info = t_rows.iloc[0]
    st.markdown("---")
    
    display_race_name = r_info.get('race_name', '')
    if pd.isna(display_race_name) or display_race_name == 'nan': display_race_name = ''
    surface_info = r_info.get('surface', '不明')
    distance_info = r_info.get('distance', '')
    
    place_name_str = r_info.get('place_name', '')
    dist_str = str(distance_info)
    dist_num_str = re.sub(r'\D', '', dist_str)
    turn_info = ""
    
    if place_name_str == "新潟" and dist_num_str == "1000":
        turn_info = "直線"
    elif place_name_str in ['中山', '阪神', '京都', '札幌', '函館', '福島', '小倉']:
        turn_info = "右回り"
    elif place_name_str in ['東京', '新潟', '中京']:
        turn_info = "左回り"
    
    turn_display = f" / {turn_info}" if turn_info else ""
    
    res_df, pat, rec_ticket, buy_detail, race_day = render_race_predictions(t_id, "良")
    
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{display_race_name}】 <span style='font-size:18px; color:#7f8c8d;'>({surface_info}{distance_info}m{turn_display} / 開催{race_day}日目)</span></h2>", unsafe_allow_html=True)
    
    cond = st.radio("想定馬場 (AIのコース・馬場バイアス補正に反映されます)", ["良", "稍重", "重", "不良"], horizontal=True)
    
    res_df, pat, rec_ticket, buy_detail, _ = render_race_predictions(t_id, cond)
    
    if res_df is not None:
        is_mokuyou = any(res_df['馬番'].astype(str).str.startswith('90'))
        if is_mokuyou:
            st.info("ℹ️ 現在は枠順確定前の「特別登録状態」のため、枠番・馬番は未定として表示され、枠順バイアスは計算から除外されています。")
            
        st.markdown(f"""
        <div class='sense-card'>
            <div class='sense-title'>🧠 AI勝負気配判定: {pat}</div>
            <div style='margin-top: 8px;'><b>🎟 推奨買い目:</b> <span class='ticket-badge'>{rec_ticket}</span></div>
            <div style='margin-top: 6px; font-size: 15px; color: #555;'><b>📝 買い目詳細:</b> {buy_detail}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Pythonベース予想データ (全要素バイアス補正済み)</div>", unsafe_allow_html=True)
        st.markdown(generate_base_table(res_df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🧠 Geminiに全データを渡し、展開予想と定性評価を実行", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()

            race_class = "条件戦"
            if re.search(r'新馬|メイクデビュー', display_race_name):
                race_class = "新馬"
            elif re.search(r'未勝利', display_race_name):
                race_class = "未勝利"
            elif re.search(r'1勝|500万', display_race_name):
                race_class = "1勝クラス"
            elif re.search(r'2勝|1000万', display_race_name):
                race_class = "2勝クラス"
            elif re.search(r'3勝|1600万', display_race_name):
                race_class = "3勝クラス"
            elif re.search(r'G1|G2|G3|OP|オープン|リステッド|L', display_race_name, re.IGNORECASE):
                race_class = "OP"
                
            waku_info_text = ""
            df_waku = load_waku_stats()
            search_surface = 'ダート' if 'ダ' in str(surface_info) else '芝'
            
            waku_extracted_name = ""
            w_data = None

            if not df_waku.empty:
                match = pd.DataFrame()
                for idx, row in df_waku.iterrows():
                    csv_race_name = str(row['レース名_クラス']).strip()
                    if csv_race_name in display_race_name and csv_race_name not in ['OP', '3勝クラス', '2勝クラス', '1勝クラス', '未勝利', '新馬']:
                        match = df_waku.iloc[[idx]]
                        waku_extracted_name = csv_race_name
                        break
                
                if match.empty:
                    match = df_waku[(df_waku['レース名_クラス'] == race_class) & 
                                    (df_waku['競馬場'] == place_name_str) & 
                                    (df_waku['芝ダート'] == search_surface) & 
                                    (df_waku['距離'] == dist_num_str)]
                    if not match.empty:
                        waku_extracted_name = f"{place_name_str}{search_surface}{dist_num_str}m ({race_class})"

                if match.empty:
                    match = df_waku[(df_waku['競馬場'] == place_name_str) & 
                                    (df_waku['芝ダート'] == search_surface) & 
                                    (df_waku['距離'] == dist_num_str)]
                    if not match.empty:
                        waku_extracted_name = f"{place_name_str}{search_surface}{dist_num_str}m (クラス不問/参考値)"

                if not match.empty:
                    w_data = match.iloc[0]
            
            if w_data is not None:
                waku_info_text = f"\n【㊙️トラックマン極秘データ：過去10年の同条件({waku_extracted_name}) 枠順別成績】\n"
                for i in range(1, 9):
                    win = w_data.get(f'{i}枠_勝率', '-')
                    pl = w_data.get(f'{i}枠_複勝率', '-')
                    waku_info_text += f"・{i}枠：勝率 {win}% / 複勝率 {pl}%\n"
                st.success(f"✅ Geminiにカンペを渡しました: 『過去10年の {waku_extracted_name} 枠順別成績』を参考にして考察します。")
            else:
                waku_info_text = f"\n【㊙️トラックマン極秘データ：過去10年の同条件({place_name_str}{search_surface}{dist_num_str}m) 枠順別成績】\n"
                for i in range(1, 9):
                    waku_info_text += f"・{i}枠：勝率 12.5% / 複勝率 37.5%\n"
                st.info(f"ℹ️ {place_name_str}{search_surface}{dist_num_str}mの枠データがCSVにないため、枠の有利不利はない(フラット)という前提でGeminiに考察させます。必要ならCSVにデータを追記してください。")

            if race_class == "新馬":
                search_instruction = "【新馬戦の特別指示】脚質データがないため、Google検索で『馬名 新馬 評判』を調べ、関係者の絶賛コメントや血統から脚質と適性を推測してください。"
            elif race_class == "OP" or re.search(r'G1|G2|G3', display_race_name, re.IGNORECASE):
                search_instruction = "【重賞・OPの特別指示】Google検索で『対象レース名 過去10年 傾向』を調べ、好走ローテと合致度を評価してください。脚質が不明な馬がいれば前走を検索して脚質を推測してください。"
            else:
                search_instruction = "【条件戦の特別指示】提供された調教タイム・評価とコースの有利不利を考慮してください。脚質が不明な馬は無理に推測せず、フラットに評価してください。"

            table_summary = []
            for _, r in res_df.iterrows():
                w_val = r.get('枠番', '-')
                w_str = w_val if pd.notna(w_val) and str(w_val) != 'nan' and not str(w_val).startswith('90') else "未定"
                
                m_val = r.get('馬番', '-')
                m_str = m_val if pd.notna(m_val) and str(m_val) != 'nan' and not str(m_val).startswith('90') else "未定"
                
                win_val = float(r.get('win_prob', 0)) * 100
                score_val = int(r.get('ai_score', 100))
                
                info = (f"枠:{w_str} | 馬番:{m_str} | 馬名:{r.get('馬名','')} | Pythonスコア:{score_val} (1着率:{win_val:.1f}%) | 騎手:{r.get('騎手','')} | 調教師:{r.get('調教師','')} | "
                        f"血統(父/母父):{r.get('父','-')} / {r.get('母父','-')} | "
                        f"脚質:{r.get('推測脚質','不明')} | 前走:{r.get('前走着順','-')} (通過:{r.get('前走通過順','-')} / 上がり3F:{r.get('前走上がり3F','-')}) | "
                        f"調教評価:{r.get('調教評価','-')} | 調教タイム:{r.get('調教タイム','-')} | 短評:{r.get('調教短評','-')} | 厩舎コメント:{r.get('厩舎コメント','-')}")
                table_summary.append(info)

            system_instruction = f"""
あなたは競馬のトラックバイアス（馬場傾向）、展開、調教タイム、血統を専門とする超一流のプロ競馬予想家（トラックマン）です。

【あなたの至上命題】
あらかじめ用意されたテンプレートに当てはめるのではなく、出走メンバーの顔ぶれから自然な展開ストーリーを組み立て、各種データ（馬場・枠・調教生時計）を掛け合わせた結果、「Pythonの基本評価以上にパフォーマンスを跳ね上げる真の好走馬」を導き出すことです。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【思考プロセスの4手順（厳守）】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

◆ Step 1：出走馬の脚質構成とコース条件から「リアルな展開ストーリー」を描く
まず出走馬リストの脚質を数え、コース特性と掛け合わせてストーリーを構築してください。
1. 逃げ馬は何頭いるか？（単騎逃げでマイペースか、複数頭でハナを争うか）
2. 先行勢は多いか？ 差し・追込ばかりか？
3. 競馬場・距離（直線の長短・坂など）と想定馬場（良・稍重・重・不良）、そして提供された枠順傾向（内外の有利不利）を加味。
➔ 「このメンバーとコース条件なら、道中のペースはこうなり、直線ではここを通る馬に最も展開が向く」という必然的な展開ストーリーを言語化してください。

◆ Step 2：「脚質×調教生時計」による仕上がりジャッジ
提供された「調教評価（A, B, C等）」は他人の主観なので完全に無視してください。
必ず【調教タイム】の生データそのものを見ること：
・逃げ・先行馬：テンの行き脚を示す「4F・5Fの全体時計」の速さを重視。
・差し・追込馬：終いのキレ味を示す「ラスト1F・2Fの加速ラップ（11秒台〜12秒前半）」を重視。
ストーリーに合致していても、時計が鈍い馬は割り引いてください。

◆ Step 3：Python評価とのギャップ分析（評価の上昇・下降）
提供されたPythonスコア（過去データや持ち時計のベース能力）と比較してください。
・【評価上昇】：Pythonスコアは控えめでも、今回のストーリー・枠利・馬場バイアス・調教生時計がドンピシャでハマり、激走する根拠が揃っている馬。
・【評価下降】：Pythonスコアは高いが、今回のストーリー展開や馬場・枠バイアスと完全に逆行している危険な人気馬。

◆ Step 4：配役（印の決定）
Step 3で導き出した「評価上昇馬」を中心に印を打ってください。
・【◎】（1頭）：ストーリー・条件バイアス・調教時計が最も噛み合い、Python評価以上に勝つ見込みが高い馬。
・【◯】（1頭）：ストーリーに合致し、本命に匹敵する対抗馬。
・【▲】（1頭）：単穴。展開や馬場がハマれば頭まで突き抜ける魅力馬。
・【△】（1頭）：連下の押さえ。
・【☆】（1〜2頭）：特注穴馬。Python下位（人気薄）であっても、ストーリーや道悪血統、生時計に明確な裏付けがある穴馬をピンポイントで抜擢。
・【消】：Python上位であっても、ストーリーに合致せず危険と判断した馬は理由を明記して切る。

※必ず以下のJSON形式のみで出力してください。Markdownタグは含めず、純粋なJSON文字列のみを返してください。
{{
  "race_view": "【展開・馬場総括】今回の出走構成は逃げ馬が不在で、先行勢も手薄。必然的にペースは緩み、前走で好位を取れている馬に絶対的有利な展開となる。さらに枠データを見ても内枠が優勢。差し馬には厳しい流れと見て...",
  "evaluations": [ 
    {{
      "馬名": "馬名", 
      "pros_cons": "【好材料】栗東坂路で自己ベストとなる全体時計51.5秒をマーク。内枠からすんなり先行できる展開利。【懸念点】重馬場の実績に乏しい。",
      "Gemini印": "◎", 
      "短評": "前有利のストーリーに最も合致する1頭。調教生時計の全体ラップが秀逸で、恵まれた展開と内枠を活かせば勝ち切れると見て評価上昇。"
    }}
  ] 
}}
"""

            prompt = f"""
【分析対象レース】
レース名: {sel_date} {r_info['place_name']} {r_info['r_num']}R {display_race_name}
コース: {surface_info} {distance_info}m {turn_display} (開催{race_day}日目)

【⚠️最重要コンディション】
想定馬場: 【 {cond} 】
Pythonによる戦況ジャッジ: 【 {pat} 】
※この戦況ジャッジと想定馬場を踏まえ、4ステップ手順に沿って分析してください。

{waku_info_text}

【出走馬リスト（Pythonスコア・調教タイム・血統・脚質・厩舎）】
{chr(10).join(table_summary)}
"""

            with st.spinner("🧠 Geminiがメンバー構成から独自の展開ストーリーを紡ぎ出しています..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                gemini_data = None
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash', 
                        contents=prompt, 
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction, 
                            temperature=0.7, 
                            tools=[{"googleSearch": {}}]
                        )
                    )
                    
                    res_text = response.text if response.text else ""
                    # ⚠️ 修正箇所: ここで文字列リテラル内の改行がエラーにならないように正規表現を修正
                    res_text = re.sub(r'^```(?:json)?\n', '', res_text)
                    res_text = re.sub(r'\n```$', '', res_text)
                    res_text = res_text.strip()
                    
                    match = re.search(r'\{.*\}', res_text, re.DOTALL)
                    if match:
                        res_text = match.group(0)
                        
                    gemini_data = json.loads(res_text)
                    
                except json.JSONDecodeError as e:
                    st.error("❌ JSONパースエラー: Geminiの出力形式が不正です。もう一度ボタンを押してください。")
                    st.write("【エラー出力の生データ】", res_text)
                except Exception as e:
                    st.error(f"❌ API通信エラー: {e}")
                    
                if gemini_data:
                    gemini_view = gemini_data.get("race_view", "")
                    
                    evals = gemini_data.get("evaluations", [])
                    eval_df = pd.DataFrame(evals)
                    
                    if not eval_df.empty and '馬名' in eval_df.columns:
                        eval_df['馬名_clean'] = eval_df['馬名'].astype(str).apply(clean_name)
                        eval_df_clean = eval_df[['馬名_clean', 'Gemini印', '短評', 'pros_cons'] if 'pros_cons' in eval_df.columns else ['馬名_clean', 'Gemini印', '短評']].dropna(subset=['馬名_clean'])
                        
                        merged_df = pd.merge(res_df, eval_df_clean, on='馬名_clean', how='left')
                        merged_df['Gemini印'] = merged_df['Gemini印'].fillna('消')
                        merged_df['短評'] = merged_df['短評'].fillna('-')
                        
                        if gemini_view:
                            st.markdown(f"""
                            <div class='gemini-view-card'>
                                <div class='gemini-view-title'>📝 トラックマンGeminiの展開・血統総合見解</div>
                                <p class='gemini-view-text'>{gemini_view}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("<div class='section-header'>🔥 Python（確率脳） × Gemini（定性・血統脳） 独立評価比較</div>", unsafe_allow_html=True)
                        st.markdown(generate_fusion_table(merged_df), unsafe_allow_html=True)