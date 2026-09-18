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
    .sense-title { font-size: 1.2rem; font-weight: bold; color: #8e44ad; }
    .ticket-badge { font-size: 1.1rem; font-weight: bold; color: #d35400; background: #fef5e7; padding: 4px 10px; border-radius: 4px; display: inline-block; }
    .gemini-win5-box { background-color: #f8f9fa; border: 2px solid #f1c40f; border-radius: 8px; padding: 20px; margin-top: 10px; line-height: 1.6; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

st.title("🐴 AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
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

@st.cache_data
def load_future_data():
    if os.path.exists(FUTURE_CSV):
        df_f = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding='utf-8-sig')
        df_f['race_id'] = df_f['race_id'].astype(str).str.zfill(12)
        df_f['place_name'] = df_f['race_id'].str[4:6].map({"01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京","06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉"}).fillna("開催場")
        df_f['r_num'] = pd.to_numeric(df_f['race_id'].str[-2:], errors='coerce').fillna(1).astype(int)
        df_f['day_label'] = df_f['date'].astype(str).str.strip() if 'date' in df_f.columns else "当日"
        df_f['馬名_clean'] = df_f['馬名'].astype(str).apply(clean_name)
        return df_f
    return pd.DataFrame()

# 🚨 @st.cache_data を使用し、リロードボタンで確実にクリアされる仕様に変更
@st.cache_data
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return joblib.load(CACHE_FILE)
        except Exception:
            return {}
    return {}

df_future = load_future_data()

# ==========================================
# 🧠 PKLからの超高速ロード & 馬場補正
# ==========================================
def render_race_predictions(race_id_target, cond):
    if df_future.empty: return None, None, None, None
    
    cache_data = load_cache()
    if not cache_data:
        st.warning("⚠️ キャッシュファイル(app_cache.pkl)が見つかりません。")
        return None, None, None, None
    
    df = df_future[df_future['race_id'].astype(str) == str(race_id_target)].copy()
    if df.empty: return None, None, None, None

    all_preds = cache_data.get('predictions', {})
    
    # キーの表記型（ゼロ埋め12桁/通常文字列/数値）に対応
    target_key = str(race_id_target).zfill(12)
    predictions = all_preds.get(target_key) or all_preds.get(str(race_id_target))
    
    if not predictions:
        st.error(f"⚠️ レースID '{target_key}' の計算済みデータが PKL 内に見つかりません。")
        return None, None, None, None

    horse_preds = predictions.get('horses', {})
    
    df['win_prob'] = df['馬名_clean'].map(lambda x: horse_preds.get(x, {}).get('win_prob', 0.1))
    df['ai_score'] = df['馬名_clean'].map(lambda x: horse_preds.get(x, {}).get('ai_score', 100))
    df['脚質'] = df['馬名_clean'].map(lambda x: horse_preds.get(x, {}).get('脚質', '-'))
    df['印'] = df['馬名_clean'].map(lambda x: horse_preds.get(x, {}).get('印', '消'))

    # 馬場状態・コースバイアス補正
    df['bias_coef'] = 1.0
    place_name = str(df['place_name'].iloc[0])

    if cond in ['重', '不良']:
        df['bias_coef'] *= df['脚質'].map(TRACK_BIAS["重・不良"]).fillna(1.0)
    elif cond == '稍重':
        df['bias_coef'] *= df['脚質'].map(TRACK_BIAS["稍重"]).fillna(1.0)
    
    if place_name in TRACK_BIAS:
        df['bias_coef'] *= df['脚質'].map(TRACK_BIAS[place_name]).fillna(1.0)

    df['win_prob'] = df['win_prob'] * df['bias_coef']
    sum_p = df['win_prob'].sum()
    if sum_p > 0: df['win_prob'] = df['win_prob'] / sum_p

    min_p, max_p = df['win_prob'].min(), df['win_prob'].max()
    if max_p > min_p:
        df['ai_score'] = (50 + (df['win_prob'] - min_p) / (max_p - min_p) * 100).round().astype(int)

    df = df.sort_values(by=['win_prob'], ascending=False).reset_index(drop=True)

    pat = predictions.get('pattern', '③ 混戦気配')
    rec_ticket = predictions.get('ticket', '3連複')
    buy_detail = predictions.get('buy_detail', '')

    return df, pat, rec_ticket, buy_detail

# ==========================================
# 📊 UIテーブル生成部
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
        style_val = str(r.get('脚質', '-')).strip()
        
        html += f"<tr><td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td>{style_val}</td><td>{kinryo_val:.1f}kg</td><td>{score_str}</td><td>{win_str}</td><td>{coef_str}</td>"
        html += f"<td><span class='badge-mark {get_badge_class(mark)}'>{mark}</span></td></tr>"
    html += "</table></div>"
    return html

def generate_fusion_table(merged_df):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>AIｽｺア</th><th>1着率</th><th>Python印</th><th>Gemini印</th><th style='text-align:left;'>Gemini短評</th></tr>"
    for _, r in merged_df.iterrows():
        win_val = float(r.get('win_prob', 0))
        win_str = f"<span style='color:#e74c3c; font-weight:bold;'>{win_val*100:.1f}%</span>" if win_val >= 0.25 else f"{win_val*100:.1f}%"
        html += f"<tr><td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td>"
        html += f"<td><b>{int(r.get('ai_score', 100))}</b></td><td>{win_str}</td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('印', '消'))}'>{r.get('印', '消')}</span></td>"
        html += f"<td><span class='badge-mark {get_badge_class(r.get('Gemini印', '消'))}'>{r.get('Gemini印', '消')}</span></td>"
        html += f"<td style='text-align:left; font-size:15px; color:#444;'>{r.get('短評', '-')}</td></tr>"
    html += "</table></div>"
    return html

# 🔧 ボタン押下時に全データをクリア
def clear_all_caches():
    st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()

# ==========================================
# 🎯 画面表示
# ==========================================
st.sidebar.button("🔄 画面リロード", on_click=clear_all_caches, use_container_width=True)
st.markdown("<div class='section-header'>🎯 レース選択</div>", unsafe_allow_html=True)

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
    st.markdown(f"<h2>🚀 {r_info['place_name']} {r_info['r_num']}R 【{r_info.get('race_name', '')}】</h2>", unsafe_allow_html=True)
    
    cond = st.radio("想定馬場 (AIのコースバイアス補正に反映されます)", ["良", "稍重", "重", "不良"], horizontal=True)
    
    res_df, pat, rec_ticket, buy_detail = render_race_predictions(t_id, cond)
    
    if res_df is not None:
        st.markdown(f"""
        <div class='sense-card'>
            <div class='sense-title'>🧠 AI勝負気配判定: {pat}</div>
            <div style='margin-top: 8px;'><b>🎟️ 推奨買い目:</b> <span class='ticket-badge'>{rec_ticket}</span></div>
            <div style='margin-top: 6px; font-size: 15px; color: #555;'><b>📝 買い目詳細:</b> {buy_detail}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Pythonベース予想データ (馬場・コース補正済み)</div>", unsafe_allow_html=True)
        st.markdown(generate_base_table(res_df), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 🔥 Gemini独立穴馬発掘機能
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