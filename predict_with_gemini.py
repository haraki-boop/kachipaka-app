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
# 🎨 アプリの基本設定とカスタムCSS
# ==========================================
st.set_page_config(page_title="AI予想 勝ちぱかくん", page_icon="🐴", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    .section-header {
        font-size: 1.3rem; font-weight: bold; color: #2c3e50;
        margin-top: 1rem; margin-bottom: 1rem;
        border-bottom: 2px solid #ecf0f1; padding-bottom: 5px;
    }
    .table-container {
        width: 100%; overflow-x: auto; -webkit-overflow-scrolling: touch;
        margin-bottom: 20px; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .kachi-table {
        width: 100%; border-collapse: collapse; margin-bottom: 0;
        font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #ffffff; white-space: nowrap;
    }
    .kachi-table thead tr { background: #fdfdfd; color: #2c3e50; font-weight: bold; border-bottom: 2px solid #eaeaea; }
    .kachi-table th { padding: 8px 10px; text-align: center; }
    .kachi-table td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #f4f4f4; color: #34495e; }
    .kachi-table tbody tr:hover td { background: #f9fbfd; }
    .badge-mark { color: #fff; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.85em; display: inline-block; min-width: 60px;}
    .badge-honmei { background: #e74c3c; }
    .badge-taikou { background: #3498db; }
    .badge-tana   { background: #2ecc71; }
    .badge-renka  { background: #f39c12; }
    .badge-ana    { background: #9b59b6; }
    .badge-keshi  { background: #e0e0e0; color: #7f8c8d; }

    @media (max-width: 768px) {
        .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }
        .kachi-table { font-size: 12px; }
        .kachi-table th, .kachi-table td { padding: 4px 6px; }
        .section-header { font-size: 1.1rem; }
        .badge-mark { min-width: 45px; padding: 2px 4px; font-size: 0.75em; }
        h1 { font-size: 1.3rem !important; } 
        h2 { font-size: 1.1rem !important; } 
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.6, 10])
with col1:
    try:
        st.image("image_61b676.png", use_container_width=True)
    except Exception:
        st.write("🐴")
with col2:
    st.title("AI予想 勝ちぱかくん")

if 'selected_race_id' not in st.session_state:
    st.session_state['selected_race_id'] = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 

FUTURE_CSV = "future_races.csv"
HISTORY_CSV = "prediction_history.csv"
ML_TARGET_CSV = "ml_target_data.csv"

# ------------------------------------------
# 🧹 馬名照合用の厳密な正規化関数
# ------------------------------------------
def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s・･.\-ー_]+', '', s).strip()

# ==========================================
# 1. データとAIモデルの読み込み
# ==========================================
@st.cache_resource
def load_model():
    for m_name in ["勝ちパカくん.pkl", "keiba_ai_model.pkl"]:
        if os.path.exists(m_name) and os.path.getsize(m_name) > 0:
            try: return joblib.load(m_name)
            except: continue
    return None

@st.cache_data
def load_past_data():
    if not os.path.exists(ML_TARGET_CSV): return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932']:
        try:
            df = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding=enc)
            if 'date' in df.columns: df = df.sort_values(by='date')
            if '馬名' in df.columns: df['馬名_clean'] = df['馬名'].astype(str).apply(clean_horse_name)
            return df
        except: continue
    return pd.DataFrame()

def load_future_data():
    if not os.path.exists(FUTURE_CSV): return pd.DataFrame()
    for enc in ['utf-8-sig', 'utf-8', 'cp932']:
        try:
            df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding=enc)
            if df.empty: continue
            if 'race_id' in df.columns:
                df['race_id'] = df['race_id'].astype(str).str.zfill(12)
                PLACE_MAP_REV = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
                df['place_code'] = df['race_id'].str[4:6]
                df['place_name'] = df['place_code'].map(PLACE_MAP_REV).fillna("開催場")
                df['r_num'] = pd.to_numeric(df['race_id'].str[10:12], errors='coerce').fillna(1).astype(int)
            
            if 'race_name' not in df.columns: df['race_name'] = ""
            else: df['race_name'] = df['race_name'].astype(str).str.replace('👑', '', regex=False).str.strip()

            if '馬名' in df.columns: df['馬名_clean'] = df['馬名'].astype(str).apply(clean_horse_name)

            if 'date' in df.columns and df['date'].notna().any(): df['day_label'] = df['date'].astype(str).str.strip()
            elif '開催日' in df.columns and df['開催日'].notna().any(): df['day_label'] = df['開催日'].astype(str).str.strip()
            else: df['day_label'] = "当日"
            return df
        except: continue
    return pd.DataFrame()

def load_history_data():
    if not os.path.exists(HISTORY_CSV): return pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay', 'pay_sanrenpuku', 'pay_sanrentan_axis', 'pay_sanrentan_form'])
    for enc in ['utf-8-sig', 'utf-8', 'cp932']:
        try:
            df = pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str, 'partners': str}, encoding=enc)
            if 'partners' not in df.columns: df['partners'] = ""
            for col in ['pay_sanrenpuku', 'pay_sanrentan_axis', 'pay_sanrentan_form']:
                if col not in df.columns: df[col] = 0
            return df
        except: continue
    return pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay', 'pay_sanrenpuku', 'pay_sanrentan_axis', 'pay_sanrentan_form'])

model_data = load_model()
df_past = load_past_data()
df_future = load_future_data()
df_history = load_history_data()

# ------------------------------------------
# 📦 過去データ辞書化
# ------------------------------------------
@st.cache_data
def build_past_horse_dict(df_p):
    if df_p.empty: return {}
    past = df_p.copy()
    past['date_parsed'] = pd.to_datetime(past['date'], errors='coerce')
    past = past.dropna(subset=['date_parsed']).sort_values(['馬名_clean', 'date_parsed'])
    
    horse_dict = {}
    for horse_clean, group in past.groupby('馬名_clean'):
        if not horse_clean: continue
        last_row = group.iloc[-1]
        
        t_vals = pd.to_numeric(group.get('my_time_idx', pd.Series()), errors='coerce').dropna()
        l_vals = pd.to_numeric(group.get('my_last3f_idx', pd.Series()), errors='coerce').dropna()
        p_vals = pd.to_numeric(group.get('my_pace_idx', pd.Series()), errors='coerce').dropna()
        s_vals = pd.to_numeric(group.get('my_start_idx', pd.Series()), errors='coerce').dropna()
        
        horse_dict[horse_clean] = {
            'last_date': last_row['date_parsed'], 
            'prev_prize': pd.to_numeric(last_row.get('賞金(万円)'), errors='coerce') if pd.notna(last_row.get('賞金(万円)')) else np.nan,
            'prev_rank': pd.to_numeric(last_row.get('着順', last_row.get('prev_rank')), errors='coerce') if pd.notna(last_row.get('着順', last_row.get('prev_rank'))) else np.nan,
            'recent3_time_idx': t_vals.tail(3).mean() if not t_vals.empty else np.nan, 
            'recent3_last3f_idx': l_vals.tail(3).mean() if not l_vals.empty else np.nan,
            'recent3_pace_idx': p_vals.tail(3).mean() if not p_vals.empty else np.nan, 
            'recent3_start_idx': s_vals.tail(3).mean() if not s_vals.empty else np.nan
        }
    return horse_dict

past_dict = build_past_horse_dict(df_past)

# ==========================================
# 2. サイドバー UI
# ==========================================
st.sidebar.header("🔄 画面の更新")
if st.sidebar.button("🔄 最新の情報にリロード", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🏁 実戦結果の検証")
if st.sidebar.button("🏆 終了したレースの配当を取得", use_container_width=True):
    with st.spinner("🏁 確定配当を検索中..."):
        if not df_history.empty:
            updated = False
            headers = {"User-Agent": "Mozilla/5.0"}
            for idx, row in df_history.iterrows():
                try:
                    axis = str(int(row['honmei_umaban']))
                    partners = [str(int(p.strip())) for p in str(row.get('partners', '')).split(',') if p.strip().isdigit()]
                    payout_found = False
                    payouts = {'三連複': 0, '三連単_軸': 0, '三連単_F': 0}
                    
                    for url in [f"https://race.netkeiba.com/race/result.html?race_id={str(row['race_id'])}", f"https://db.netkeiba.com/race/{str(row['race_id'])}/"]:
                        res = requests.get(url, headers=headers, timeout=5)
                        res.encoding = 'EUC-JP' if 'db.netkeiba' in url else 'utf-8'
                        soup = BeautifulSoup(res.text, "html.parser")
                        for tr in soup.find_all("tr"):
                            th = tr.find("th")
                            if th:
                                tds = tr.find_all("td")
                                if len(tds) < 2: continue
                                w_td, a_td = tds[0], tds[1]
                            else:
                                th = tr.find("td", class_=re.compile(r"(Result_Pay_Type|Pay_Type|type)"))
                                if not th: continue
                                tds = tr.find_all("td")
                                if len(tds) < 3: continue
                                w_td, a_td = tds[1], tds[2]
                                
                            kind_raw = th.get_text(strip=True).replace(" ", "").replace(" ", "")
                            kind = ""
                            if "3連複" in kind_raw or "３連複" in kind_raw or "三連複" in kind_raw: kind = "三連複"
                            elif "3連単" in kind_raw or "３連単" in kind_raw or "三連単" in kind_raw: kind = "三連単"
                            if not kind: continue
                            
                            for br in w_td.find_all('br'): br.replace_with(' _SPLIT_ ')
                            for ul in w_td.find_all('ul'): ul.insert_before(' _SPLIT_ ')
                            for br in a_td.find_all('br'): br.replace_with(' _SPLIT_ ')
                            for ul in a_td.find_all('ul'): ul.insert_before(' _SPLIT_ ')
                            w_lines = [s for s in w_td.get_text(separator=" ").split('_SPLIT_') if s.strip()]
                            a_lines = [s for s in a_td.get_text(separator=" ").split('_SPLIT_') if s.strip()]
                            
                            for w_str, a_str in zip(w_lines, a_lines):
                                amt_str = re.sub(r'\D', '', a_str)
                                if not amt_str.isdigit(): continue
                                amt = int(amt_str)
                                w_nums = [str(int(n)) for n in re.findall(r'\d+', w_str)]
                                if not w_nums: continue
                                payout_found = True

                                if kind == "三連複" and len(w_nums) >= 3:
                                    if all(n in ([axis] + partners) for n in w_nums[:3]): payouts['三連複'] += amt
                                elif kind == "三連単" and len(w_nums) >= 3:
                                    if w_nums[0] == axis and w_nums[1] in partners and w_nums[2] in partners: payouts['三連単_軸'] += amt
                                    form_2nd = partners[:2] if len(partners) >= 2 else partners
                                    if w_nums[0] == axis and w_nums[1] in form_2nd and w_nums[2] in partners: payouts['三連単_F'] += amt

                        if payout_found:
                            df_history.at[idx, 'pay_sanrenpuku'] = payouts['三連複']
                            df_history.at[idx, 'pay_sanrentan_axis'] = payouts['三連単_軸']
                            df_history.at[idx, 'pay_sanrentan_form'] = payouts['三連単_F']
                            df_history.at[idx, 'result_pay'] = payouts['三連複'] + payouts['三連単_軸']
                            updated = True
                            break
                except: pass
                time.sleep(1)
            
            if updated: df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig', errors='replace')
        st.cache_data.clear()
        st.success("✅ 実戦結果を最新化しました！")
        time.sleep(1.5)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("🗑️ 履歴の完全リセット")
if st.sidebar.button("💥 予想履歴を消去", type="primary", use_container_width=True):
    try:
        if os.path.exists(HISTORY_CSV): os.remove(HISTORY_CSV)
        st.cache_data.clear()
        st.sidebar.success("✅ 履歴データを消去しました！")
        time.sleep(1.5)
        st.rerun()
    except: pass

# ==========================================
# 3. AIスコア算出（★実力主義・嘘データ補填排除版★）
# ==========================================
def calculate_race_scores(race_id_target, target_df, user_condition="良"):
    if target_df.empty or model_data is None: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None

    features = model_data.get('features', [])
    model = model_data.get('model')

    if '馬名_clean' not in race_df.columns:
        race_df['馬名_clean'] = race_df['馬名'].astype(str).apply(clean_horse_name)

    raw_odds = pd.to_numeric(race_df.get('単勝', race_df.get('オッズ', pd.Series())), errors='coerce')
    race_df['単勝_num'] = raw_odds
    if '人気' not in race_df.columns or race_df['人気'].isna().all():
        race_df['人気'] = race_df['単勝_num'].rank(method='min')
    
    race_df['pop_num'] = pd.to_numeric(race_df['人気'], errors='coerce')

    def get_past_stat(horse_clean, key):
        return past_dict.get(horse_clean, {}).get(key, np.nan)
    
    # 特徴量の取得（欠損はNaNのまま）
    race_df['eff_time_idx'] = race_df['馬名_clean'].apply(lambda x: get_past_stat(x, 'recent3_time_idx'))
    race_df['eff_last3f_idx'] = race_df['馬名_clean'].apply(lambda x: get_past_stat(x, 'recent3_last3f_idx'))
    race_df['eff_pace_idx'] = race_df['馬名_clean'].apply(lambda x: get_past_stat(x, 'recent3_pace_idx'))
    race_df['eff_start_idx'] = race_df['馬名_clean'].apply(lambda x: get_past_stat(x, 'recent3_start_idx'))
    race_df['prev_prize'] = race_df['馬名_clean'].apply(lambda x: get_past_stat(x, 'prev_prize'))
    race_df['prev_rank_num'] = race_df['馬名_clean'].apply(lambda x: get_past_stat(x, 'prev_rank'))
    
    race_df['date_parsed_fut'] = pd.to_datetime(race_df['date'], errors='coerce')
    race_df['last_date'] = pd.to_datetime(race_df['馬名_clean'].apply(lambda x: get_past_stat(x, 'last_date')), errors='coerce')
    race_df['interval_days'] = (race_df['date_parsed_fut'] - race_df['last_date']).dt.days

    j_col = race_df.get('jockey_win_power', race_df.get('jockey_win_rate', pd.Series()))
    race_df['eff_jockey_win'] = pd.to_numeric(j_col, errors='coerce').clip(0.0, 1.0)
    race_df['eff_jockey_track_win'] = pd.to_numeric(race_df.get('jockey_track_win_rate'), errors='coerce').clip(0.0, 1.0)
    race_df['horse_win_rate_val'] = pd.to_numeric(race_df.get('horse_win_rate'), errors='coerce').clip(0.0, 1.0)
    race_df['horse_runs_val'] = pd.to_numeric(race_df.get('horse_runs'), errors='coerce')

    race_df['condition'] = user_condition
    race_df['condition_code'] = user_condition

    X = pd.DataFrame(index=race_df.index)
    for f in features:
        if f in race_df.columns:
            X[f] = race_df[f]
        else:
            X[f] = np.nan

    cat_cols = ['surface_code', 'condition_code', 'sex_code']
    for cat in cat_cols:
        if cat in X.columns:
            X[cat] = X[cat].astype('category')
            if cat == 'sex_code': X[cat] = X[cat].replace({'牡': 0, '牝': 1, 'セ': 2})
            if cat == 'surface_code': X[cat] = X[cat].replace({'芝': 0, 'ダート': 1, '障害': 2})
            if cat == 'condition_code': X[cat] = X[cat].replace({'良': 0, '稍重': 1, '稍': 1, '重': 2, '不良': 3})

    X_num = X.copy()
    for col in X_num.columns:
        if col not in cat_cols:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

    try:
        raw_scores = model.predict(X_num)
    except Exception as e:
        st.error(f"モデル予測エラー: {e}")
        return None

    race_df['win_prob'] = raw_scores

    # スコア化（偏差値）
    std_p = np.std(race_df['win_prob'])
    mean_p = np.mean(race_df['win_prob'])
    if pd.isna(std_p) or std_p < 1e-5: 
        race_df['score_brain1'] = 50
    else: 
        race_df['score_brain1'] = (50 + (race_df['win_prob'] - mean_p) / std_p * 10).round().astype(int)

    race_df['ev_brain2'] = race_df['win_prob'] * race_df['単勝_num']

    # 脚質判定
    total_horses = len(race_df)
    if total_horses > 0 and race_df['eff_start_idx'].notna().any():
        race_df['start_rank'] = race_df['eff_start_idx'].rank(ascending=False, method='min')
        def determine_style(row):
            if pd.isna(row.get('start_rank')): return "-" 
            pct = row['start_rank'] / total_horses
            if pct <= 0.15: return "逃げ"
            elif pct <= 0.40: return "先行"
            elif pct <= 0.75: return "差し"
            else: return "追込"
        race_df['脚質'] = race_df.apply(determine_style, axis=1)
    else:
        race_df['脚質'] = "-"

    # ★実力主義：AIスコア順でソート
    pure_sorted = race_df.sort_values(by=['score_brain1', 'pop_num'], ascending=[False, True]).reset_index(drop=True)
    race_df['印'] = "消"
    
    if len(pure_sorted) > 0: race_df.loc[race_df['馬番'] == pure_sorted.loc[0, '馬番'], '印'] = "◎ 本命"
    if len(pure_sorted) > 1: race_df.loc[race_df['馬番'] == pure_sorted.loc[1, '馬番'], '印'] = "◯ 対抗"

    unmarked_mask = race_df['印'] == "消"
    if unmarked_mask.any():
        remaining_df = race_df[unmarked_mask].copy()
        remaining_df['紐評価'] = remaining_df['score_brain1'].fillna(0) + (remaining_df['ev_brain2'].fillna(0) * 5)
        himo_sorted_idx = remaining_df.sort_values(by=['紐評価', 'score_brain1'], ascending=[False, False]).index
        
        if len(himo_sorted_idx) > 0: race_df.loc[himo_sorted_idx[0], '印'] = "▲ 単穴"
        if len(himo_sorted_idx) > 1: race_df.loc[himo_sorted_idx[1], '印'] = "△ 連下"
        if len(himo_sorted_idx) > 2: race_df.loc[himo_sorted_idx[2], '印'] = "△ 連下"
        
        if len(himo_sorted_idx) > 3:
            for idx in himo_sorted_idx[3:]:
                if race_df.loc[idx, '単勝_num'] >= 10.0:
                    race_df.loc[idx, '印'] = "☆ 穴馬"
                    break

    mark_order = {"◎ 本命": 1, "◯ 対抗": 2, "▲ 単穴": 3, "△ 連下": 4, "☆ 穴馬": 5, "消": 6}
    race_df['mark_rank'] = race_df['印'].map(mark_order).fillna(99)
    race_df = race_df.sort_values(by=['mark_rank', 'score_brain1'], ascending=[True, False]).reset_index(drop=True)

    return race_df

# ==========================================
# 4. テーブル生成
# ==========================================
def generate_beautiful_table(disp_df, is_newcomer):
    html = "<div class='table-container'><table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>騎手</th><th>脚質</th><th>AIスコア</th><th>複勝確率</th><th>予想オッズ</th><th>期待値</th><th>印</th></tr></thead><tbody>"
    
    for i, r in disp_df.iterrows():
        ev_val = float(r.get('ev_brain2', 0)) if pd.notna(r.get('ev_brain2')) else 0.0
        odds_val = float(pd.to_numeric(r.get('単勝', r.get('オッズ', 0)), errors='coerce')) if pd.notna(r.get('単勝', r.get('オッズ', 0))) else 0.0
        win_prob_val = float(r.get('win_prob', 0)) if pd.notna(r.get('win_prob')) else 0.0
        mark = r.get('印', '消')
        kyakushitsu = r.get('脚質', '-')
        
        badge_cls = "badge-keshi"
        if "◎" in mark: badge_cls = "badge-honmei"
        elif "◯" in mark: badge_cls = "badge-taikou"
        elif "▲" in mark: badge_cls = "badge-tana"
        elif "△" in mark: badge_cls = "badge-renka"
        elif "☆" in mark: badge_cls = "badge-ana"
        
        score_str = f"<b>{int(r.get('score_brain1', 50))}</b>" if not is_newcomer else "-"
        win_str = f"<b>{win_prob_val*100:.1f}%</b>" if win_prob_val > 0 else "-"
        odds_str = f"{odds_val:.1f}倍" if odds_val > 0 else "-"
        ev_str = f"<b>{ev_val:.2f}</b>" if ev_val > 0 and not is_newcomer else "-"
        
        html += f"<tr><td><b>{int(r['馬番']):02d}</b></td><td style='text-align:left; font-weight:bold;'>{r.get('馬名', '-')}</td><td>{r.get('騎手', '-')}</td><td>{kyakushitsu}</td><td>{score_str}</td><td>{win_str}</td><td>{odds_str}</td><td>{ev_str}</td><td><span class='badge-mark {badge_cls}'>{mark}</span></td></tr>"
        
    html += "</tbody></table></div>"
    return html

# ==========================================
# 5. メインUI (タブ構成)
# ==========================================
tab_forecast, tab_dashboard = st.tabs(["🏇 レース予想", "📈 実戦成績"])

with tab_forecast:
    if df_future.empty:
        st.warning("⚠️ 出馬表データが存在しません。")
    else:
        st.markdown("<div class='section-header'>🎯 予想レースを選択</div>", unsafe_allow_html=True)
        date_options = sorted(df_future['day_label'].unique())
        selected_date = st.radio("開催日", date_options, horizontal=True, label_visibility="collapsed")
        day_df = df_future[df_future['day_label'] == selected_date]
        places = day_df['place_name'].unique()
        
        place_tabs = st.tabs([f"📍 {p}" for p in places])
        for p_idx, place in enumerate(places):
            with place_tabs[p_idx]:
                place_df = day_df[day_df['place_name'] == place]
                races = sorted(place_df['r_num'].unique())
                
                for i in range(0, len(races), 6):
                    chunk = races[i:i+6]
                    cols = st.columns(6)
                    for j, r in enumerate(chunk):
                        col = cols[j]
                        race_rows = place_df[place_df['r_num'] == r]
                        rid = race_rows['race_id'].iloc[0]
                        
                        # 🚨 IndexErrorを修正した箇所
                        rname = str(race_rows['race_name'].iloc[0]).strip() if 'race_name' in race_rows.columns and pd.notna(race_rows['race_name'].iloc[0]) else ""
                        label = f"{r}R {rname}".strip() if rname and rname != "nan" else f"{r}R".strip()
                            
                        if col.button(label, key=f"btn_{rid}", use_container_width=True, type="secondary"):
                            st.session_state['selected_race_id'] = rid

    if st.session_state['selected_race_id'] and not df_future.empty:
        st.markdown("---")
        target_id = str(st.session_state['selected_race_id'])
        target_rows = df_future[df_future['race_id'].astype(str) == target_id]
        
        if target_rows.empty:
            st.session_state['selected_race_id'] = None
            st.rerun()
            
        target_race_info = target_rows.iloc[0]
        rname = target_race_info.get('race_name', "")
        is_newcomer = "新馬" in str(rname)
        race_display_name = f"{target_race_info['place_name']} {target_race_info['r_num']}R 【{rname}】"
        st.markdown(f"<h2>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        selected_condition = st.radio("想定馬場", ["良", "稍重", "重", "不良"], horizontal=True)
        scored_df = calculate_race_scores(target_id, df_future, selected_condition)
        
        if scored_df is not None:
            st.markdown("<div class='section-header'>📊 出走馬 期待値＆データ一覧</div>", unsafe_allow_html=True)
            disp_df = scored_df.copy()
            if is_newcomer: st.info("🐣 新馬戦のため、過去データが存在せずベースAIスコアは参考値です。")
            
            sort_option = st.selectbox("🔄 テーブルの表示順", ["システム推奨（印順）", "馬番順（昇順）", "AIスコア順（高い順）", "複勝確率順（高い順）", "予想オッズ順（低い順）", "期待値順（高い順）"], index=0)

            if sort_option == "馬番順（昇順）": disp_df = disp_df.sort_values(by='馬番')
            elif sort_option == "AIスコア順（高い順）": disp_df = disp_df.sort_values(by=['score_brain1', '馬番'], ascending=[False, True])
            elif sort_option == "複勝確率順（高い順）": disp_df = disp_df.sort_values(by=['win_prob', '馬番'], ascending=[False, True])
            elif sort_option == "予想オッズ順（低い順）": disp_df = disp_df.sort_values(by=['単勝_num', '馬番'], ascending=[True, True])
            elif sort_option == "期待値順（高い順）": disp_df = disp_df.sort_values(by=['ev_brain2', '馬番'], ascending=[False, True])

            st.markdown(generate_beautiful_table(disp_df, is_newcomer), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 Geminiで最終適正化＆買い目生成", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
            if scored_df is None or len(scored_df) < 6:
                st.error("データが不足しているため予想をスキップします。")
                st.stop()

            top_odds = disp_df['単勝_num'].min()
            prob_gap = disp_df['win_prob'].iloc[0] - disp_df['win_prob'].iloc[1] if len(disp_df) > 1 else 0
            race_type = "【固】※本命信頼・3連単紐荒れ狙い" if (top_odds <= 2.5 or prob_gap >= 0.15) else "【荒】※上位拮抗・三連複5頭BOX/手広くフォーメーション狙い"

            table_summary = []
            for idx, row in disp_df.iterrows():
                ev_val = float(row.get('ev_brain2', 0)) if pd.notna(row.get('ev_brain2')) else 0.0
                odds_val = float(pd.to_numeric(row.get('単勝', row.get('オッズ', 0)), errors='coerce')) if pd.notna(row.get('単勝', row.get('オッズ', 0))) else 0.0
                table_summary.append(f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | 脚質:{row.get('脚質', '不明')} | オッズ:{odds_val}倍 | 複勝AIスコア:{row.get('score_brain1', 0)} | 期待値:{ev_val:.2f} | システム評価:{row.get('印', '消')}")

            system_instruction = f"""あなたはプロの競馬分析AIです。
【判定】{race_type}
【指示】実力最上位を軸にし、紐穴を厳選してください。単勝・馬連は出力せず、3連複・3連単の買い目のみ出力すること。
フォーマットは以下を守ること:
### 🌪️ レース判定・展開考察
### 💥 勝ちぱかくんの最終ジャッジ（印と根拠）
◎（本命）: 〇〇番（馬名）
◯（対抗）: 〇〇番（馬名）
▲（単穴）: 〇〇番（馬名）
△（連下）: 〇〇番、〇〇番
☆（穴馬）: 〇〇番
### 💡 戦略的・推奨全買い目
"""
            prompt = f"対象レース: {selected_date} {race_display_name}\n【想定馬場状態】: {selected_condition}\n\n出走馬データ:\n{chr(10).join(table_summary)}"

            with st.spinner("AIが検索・分析中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                res_text = ""
                for _ in range(3):
                    try:
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.3, tools=[{"googleSearch": {}}]))
                        res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                        if res_text: break
                    except: time.sleep(3)
                    
                if res_text:
                    st.markdown(res_text)
                    honmei_match = re.search(r'◎[^\d]*(\d+)番', res_text)
                    h_umaban = int(honmei_match.group(1)) if honmei_match else int(disp_df.iloc[0]['馬番'])
                    
                    partners_list = []
                    for symbol in ['◯', '▲', '☆']:
                        m = re.search(rf'{symbol}[^\n]*?(\d+)番', res_text)
                        if m: partners_list.append(m.group(1))
                    renka_line = re.search(r'△[^\n]*', res_text)
                    if renka_line: partners_list.extend(re.findall(r'(\d+)番', renka_line.group(0)))
                        
                    clean_partners = ",".join([p for p in dict.fromkeys(partners_list) if int(p) != h_umaban])
                    
                    if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                        new_record = pd.DataFrame([{'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'race_id': str(target_id), 'race_name': race_display_name, 'honmei_umaban': h_umaban, 'partners': clean_partners, 'honmei_name': "履歴参照", 'result_pay': "", 'pay_sanrenpuku': 0, 'pay_sanrentan_axis': 0, 'pay_sanrentan_form': 0}])
                        df_history = pd.concat([df_history, new_record], ignore_index=True)
                        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
                    st.success("📝 実戦履歴に記録しました！")

# ==========================================
# 6. ダッシュボード
# ==========================================
with tab_dashboard:
    st.markdown("<div class='section-header'>📈 実戦成績ダッシュボード (3連系検証)</div>", unsafe_allow_html=True)
    if df_history.empty: 
        st.info("まだ予想履歴がありません。")
    else:
        df_history['result_pay'] = df_history['result_pay'].replace(['None', 'nan', 'NaN', ''], np.nan)
        df_history['datetime'] = pd.to_datetime(df_history['date'], errors='coerce')
        df_history['year_month'] = df_history['datetime'].dt.strftime('%Y年%m月')
        df_history['just_date'] = df_history['datetime'].dt.strftime('%Y-%m-%d')
        
        tab_total, tab_month, tab_day = st.tabs(["🏆 総合成績", "📅 月別成績", "📆 日別成績"])
        
        def render_dashboard_for_df(raw_df, title_prefix):
            total_races = len(raw_df)
            if total_races == 0: return
            
            finished_df = raw_df[pd.to_numeric(raw_df['result_pay'], errors='coerce').notna()]
            pending_df = raw_df[pd.to_numeric(raw_df['result_pay'], errors='coerce').isna()]
            total = len(finished_df)
            
            if total == 0:
                st.markdown(f"**{title_prefix} 確定レース**: 0 件 （結果待ち: {total_races} 件）")
                return

            hits = len(finished_df[pd.to_numeric(finished_df['result_pay'], errors='coerce') > 0])
            returns = pd.to_numeric(finished_df['result_pay'], errors='coerce').sum()
            
            invested_total = inv_sanrenpuku = inv_sanrentan_axis = inv_sanrentan_form = 0
            for _, r in finished_df.iterrows():
                p_list = [x for x in str(r.get('partners', '')).split(',') if x.strip().isdigit()]
                p_len = len(p_list)
                if p_len > 0:
                    box_count = p_len + 1
                    if box_count >= 3: inv_sanrenpuku += int(box_count * (box_count - 1) * (box_count - 2) / 6) * 100
                    if p_len >= 2:
                        inv_sanrentan_axis += (p_len * (p_len - 1)) * 100
                        inv_sanrentan_form += (2 * (p_len - 1)) * 100
            
            invested_total = inv_sanrenpuku + inv_sanrentan_axis
            roi_total = (returns / invested_total) * 100 if invested_total > 0 else 0.0
            profit_total = int(returns - invested_total)
            
            st.markdown(f"**{title_prefix} 確定レース**: {total} 件 （結果待ち: {len(pending_df)} 件）")
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 的中率", f"{(hits/total)*100:.1f}%")
            col2.metric("💰 回収率", f"{roi_total:.1f}%")
            col3.metric("💴 収支", f"{profit_total:,} 円")

        with tab_total: render_dashboard_for_df(df_history, "総合")