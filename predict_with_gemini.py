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
    
    /* タイトル・見出しのベースサイズ */
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 5px;
    }
    
    /* 表の横スクロール用コンテナ（スマホ対応） */
    .table-container {
        width: 100%;
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        margin-bottom: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    
    .kachi-table {
        width: 100%; border-collapse: collapse; margin-bottom: 0;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        background-color: #ffffff;
        white-space: nowrap; /* スマホで改行されすぎないようにする */
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

    /* スマホ画面向けのレスポンシブデザイン */
    @media (max-width: 768px) {
        .block-container { padding-top: 1rem; padding-bottom: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }
        .kachi-table { font-size: 12px; }
        .kachi-table th, .kachi-table td { padding: 4px 6px; }
        .section-header { font-size: 1.1rem; }
        .badge-mark { min-width: 45px; padding: 2px 4px; font-size: 0.75em; }
        
        /* 📱追加：スマホでの文字サイズ縮小＆改行防止 */
        h1 { font-size: 1.3rem !important; } /* 「AI予想 勝ちぱかくん」を小さく */
        h2 { font-size: 1.1rem !important; } /* 🚀〇〇 1R... を小さく */
        
        /* 📱追加：レース選択ボタンの文字を極小にし、1行に強制 */
        .stButton button p {
            font-size: 0.65rem !important; /* ボタンの文字を極小に */
            white-space: nowrap !important; /* 強制的に1行にする */
            letter-spacing: -0.5px; /* 文字間隔を少し詰める */
        }
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

# ==========================================
# 🔑 APIキーの設定
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") 

HISTORY_CSV = "prediction_history.csv"
FUTURE_CSV = "future_races.csv"
ML_TARGET_CSV = "ml_target_data.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    return re.sub(r'[\s\u3000]+', '', s)

# ==========================================
# 1. データとAIモデルの読み込み
# ==========================================
@st.cache_resource
def load_model():
    if os.path.exists("keiba_ai_model.pkl") and os.path.getsize("keiba_ai_model.pkl") > 0:
        try:
            return joblib.load("keiba_ai_model.pkl")
        except Exception:
            return None
    return None

@st.cache_data
def load_past_data():
    if os.path.exists(ML_TARGET_CSV) and os.path.getsize(ML_TARGET_CSV) > 0:
        for enc in ['utf-8-sig', 'cp932', 'utf-8', 'shift_jis']:
            try:
                df = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding=enc)
                if 'date' in df.columns:
                    df = df.sort_values(by='date')
                if '馬名' in df.columns:
                    df['馬名_clean'] = df['馬名'].astype(str).apply(clean_horse_name)
                return df
            except Exception:
                continue
    return pd.DataFrame()

def load_future_data():
    if os.path.exists(FUTURE_CSV) and os.path.getsize(FUTURE_CSV) > 0:
        for enc in ['utf-8-sig', 'cp932', 'utf-8']:
            try:
                df = pd.read_csv(FUTURE_CSV, dtype={'race_id': str}, encoding=enc)
                PLACE_MAP_REV = {
                    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
                    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
                }
                if 'race_id' in df.columns:
                    df['place_code'] = df['race_id'].str[4:6]
                    df['place_name'] = df['place_code'].map(PLACE_MAP_REV).fillna("不明")
                    df['r_num'] = df['race_id'].str[10:12].astype(int)
                
                if 'race_name' not in df.columns: df['race_name'] = ""
                df['day_label'] = df['date'] if 'date' in df.columns else "不明"
                return df
            except Exception:
                continue
    return pd.DataFrame()

def load_history_data():
    if os.path.exists(HISTORY_CSV) and os.path.getsize(HISTORY_CSV) > 0:
        try:
            df = pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str, 'partners': str}, encoding='utf-8-sig')
        except Exception:
            try:
                df = pd.read_csv(HISTORY_CSV, dtype={'race_id': str, 'honmei_umaban': str, 'partners': str}, encoding='cp932')
            except Exception:
                return pd.DataFrame()
        if 'partners' not in df.columns:
            df['partners'] = ""
        for col in ['pay_tansho', 'pay_umaren', 'pay_wide', 'pay_sanrenpuku', 'pay_sanrentan_axis', 'pay_sanrentan_form']:
            if col not in df.columns:
                df[col] = 0
        return df
    return pd.DataFrame(columns=['date', 'race_id', 'race_name', 'honmei_umaban', 'partners', 'honmei_name', 'result_pay', 'pay_tansho', 'pay_umaren', 'pay_wide', 'pay_sanrenpuku', 'pay_sanrentan_axis', 'pay_sanrentan_form'])

model_data = load_model()
df_past = load_past_data()
df_future = load_future_data()
df_history = load_history_data()

@st.cache_data
def build_past_horse_dict(df_p):
    if df_p.empty: return {}
    past = df_p.copy()
    past['date_parsed'] = pd.to_datetime(past['date'], errors='coerce')
    past = past.dropna(subset=['date_parsed']).sort_values(['馬名', 'date_parsed'])
    
    horse_dict = {}
    for horse, group in past.groupby('馬名'):
        last_date = group['date_parsed'].iloc[-1]
        prize = pd.to_numeric(group['賞金(万円)'].iloc[-1], errors='coerce') if '賞金(万円)' in group.columns else 0.0
        t3 = pd.to_numeric(group.get('my_time_idx'), errors='coerce').tail(3).mean()
        l3 = pd.to_numeric(group.get('my_last3f_idx'), errors='coerce').tail(3).mean()
        p3 = pd.to_numeric(group.get('my_pace_idx'), errors='coerce').tail(3).mean()
        s3 = pd.to_numeric(group.get('my_start_idx'), errors='coerce').tail(3).mean()
        
        horse_dict[horse] = {
            'last_date': last_date, 'prev_prize': prize,
            'recent3_time_idx': t3, 'recent3_last3f_idx': l3,
            'recent3_pace_idx': p3, 'recent3_start_idx': s3
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
    with st.spinner("🏁 実際の着順と全券種の配当をリアルタイム検索中..."):
        if not df_history.empty:
            updated = False
            headers = {"User-Agent": "Mozilla/5.0"}
            
            # 全てのレースを最後まで回すためのループ
            for idx, row in df_history.iterrows():
                try:
                    axis = str(int(row['honmei_umaban']))
                    partners = [str(int(p.strip())) for p in str(row.get('partners', '')).split(',') if p.strip().isdigit()]
                    payout_found = False
                    payouts = {'単勝': 0, '馬連': 0, 'ワイド': 0, '三連複': 0, '三連単_軸': 0, '三連単_F': 0}
                    
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
                            if "単勝" in kind_raw: kind = "単勝"
                            elif "馬連" in kind_raw: kind = "馬連"
                            elif "ワイド" in kind_raw: kind = "ワイド"
                            elif "3連複" in kind_raw or "３連複" in kind_raw or "三連複" in kind_raw: kind = "三連複"
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

                                # 【修正】馬連やワイドなども「選んだ相手全頭」を対象に判定するよう統一
                                if kind == "単勝" and len(w_nums) >= 1 and w_nums[0] == axis:
                                    payouts['単勝'] += amt
                                elif kind == "馬連" and len(w_nums) >= 2:
                                    if axis in w_nums[:2] and any(p in w_nums[:2] for p in partners):
                                        payouts['馬連'] += amt
                                elif kind == "ワイド" and len(w_nums) >= 2:
                                    if axis in w_nums[:2] and any(p in w_nums[:2] for p in partners):
                                        payouts['ワイド'] += amt
                                elif kind == "三連複" and len(w_nums) >= 3:
                                    # 軸馬と相手馬のボックス（全通り）の中に上位3頭が含まれていれば的中
                                    if all(n in ([axis] + partners) for n in w_nums[:3]):
                                        payouts['三連複'] += amt
                                elif kind == "三連単" and len(w_nums) >= 3:
                                    if w_nums[0] == axis and w_nums[1] in partners and w_nums[2] in partners:
                                        payouts['三連単_軸'] += amt
                                    form_2nd = partners[:2] if len(partners) >= 2 else partners
                                    if w_nums[0] == axis and w_nums[1] in form_2nd and w_nums[2] in partners:
                                        payouts['三連単_F'] += amt

                        # 【修正】結果を見つけたら「URL探しのループ」だけを抜ける（次のレースの処理へ進む）
                        if payout_found:
                            df_history.at[idx, 'pay_tansho'] = payouts['単勝']
                            df_history.at[idx, 'pay_umaren'] = payouts['馬連']
                            df_history.at[idx, 'pay_wide'] = payouts['ワイド']
                            df_history.at[idx, 'pay_sanrenpuku'] = payouts['三連複']
                            df_history.at[idx, 'pay_sanrentan_axis'] = payouts['三連単_軸']
                            df_history.at[idx, 'pay_sanrentan_form'] = payouts['三連単_F']
                            df_history.at[idx, 'result_pay'] = payouts['単勝'] + payouts['馬連'] + payouts['ワイド'] + payouts['三連複'] + payouts['三連単_軸']
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
# 3. AIスコア算出
# ==========================================
def calculate_race_scores(race_id_target, target_df):
    if target_df.empty or model_data is None: return None
    race_df = target_df[target_df['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None

    features = model_data.get('features', [])
    model = model_data.get('model')

    raw_odds = pd.to_numeric(race_df.get('単勝', race_df.get('オッズ', pd.Series())), errors='coerce')
    race_df['単勝_num'] = raw_odds.fillna(15.0)
    if '人気' not in race_df.columns or race_df['人気'].isna().all():
        race_df['人気'] = race_df['単勝_num'].rank(method='min').fillna(99.0)
    
    race_df['log_odds'] = np.log(race_df['単勝_num'].clip(lower=1.1))
    race_df['pop_num'] = pd.to_numeric(race_df['人気'], errors='coerce').fillna(99.0)
    race_df['prev_rank_num'] = pd.to_numeric(race_df.get('prev_rank'), errors='coerce').fillna(9.0)

    def get_past_stat(horse_name, key, default=np.nan):
        return past_dict.get(horse_name, {}).get(key, default)
    
    race_df['recent3_time_idx'] = race_df['馬名'].apply(lambda x: get_past_stat(x, 'recent3_time_idx'))
    race_df['recent3_last3f_idx'] = race_df['馬名'].apply(lambda x: get_past_stat(x, 'recent3_last3f_idx'))
    race_df['recent3_pace_idx'] = race_df['馬名'].apply(lambda x: get_past_stat(x, 'recent3_pace_idx'))
    race_df['recent3_start_idx'] = race_df['馬名'].apply(lambda x: get_past_stat(x, 'recent3_start_idx'))
    race_df['prev_prize'] = race_df['馬名'].apply(lambda x: get_past_stat(x, 'prev_prize', 0.0))
    
    race_df['date_parsed_fut'] = pd.to_datetime(race_df['date'], errors='coerce')
    race_df['last_date'] = race_df['馬名'].apply(lambda x: get_past_stat(x, 'last_date', pd.NaT))
    race_df['interval_days'] = (race_df['date_parsed_fut'] - race_df['last_date']).dt.days.fillna(60.0)

    race_df['eff_time_idx'] = pd.to_numeric(race_df.get('recent3_time_idx'), errors='coerce').fillna(75.0)
    race_df['eff_last3f_idx'] = pd.to_numeric(race_df.get('recent3_last3f_idx'), errors='coerce').fillna(75.0)
    race_df['eff_pace_idx'] = pd.to_numeric(race_df.get('recent3_pace_idx'), errors='coerce').fillna(75.0)
    race_df['eff_start_idx'] = pd.to_numeric(race_df.get('recent3_start_idx'), errors='coerce').fillna(85.0)

    j_col = race_df.get('jockey_win_power', race_df.get('jockey_win_rate', pd.Series()))
    race_df['eff_jockey_win'] = pd.to_numeric(j_col, errors='coerce').fillna(0.05).clip(0.0, 1.0)
    race_df['eff_jockey_track_win'] = pd.to_numeric(race_df.get('jockey_track_win_rate'), errors='coerce').fillna(0.05).clip(0.0, 1.0)
    race_df['horse_win_rate_val'] = pd.to_numeric(race_df.get('horse_win_rate'), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    race_df['horse_runs_val'] = pd.to_numeric(race_df.get('horse_runs'), errors='coerce').fillna(0.0)
    race_df['course_avg_time_val'] = pd.to_numeric(race_df.get('course_avg_time'), errors='coerce').fillna(100.0)

    for orig_c, z_c in [('eff_time_idx', 'z_time_idx'), ('eff_last3f_idx', 'z_last3f_idx'), ('log_odds', 'z_odds')]:
        std_val = race_df[orig_c].std()
        if pd.isna(std_val) or std_val < 1e-5:
            race_df[z_c] = 0.0
        else:
            race_df[z_c] = (race_df[orig_c] - race_df[orig_c].mean()) / std_val

    for f in features:
        if f not in race_df.columns:
            race_df[f] = 0.0

    X = race_df[features].copy()

    if 'sex_code' in X.columns:
        X['sex_code'] = X['sex_code'].replace({'牡': 0, '牝': 1, 'セ': 2})
    if 'surface' in X.columns:
        X['surface'] = X['surface'].replace({'芝': 0, 'ダート': 1, '障害': 2})
    if 'surface_code' in X.columns:
        X['surface_code'] = X['surface_code'].replace({'芝': 0, 'ダート': 1, '障害': 2})
    if 'condition' in X.columns:
        X['condition'] = X['condition'].replace({'良': 0, '稍重': 1, '稍': 1, '重': 2, '不良': 3})
    if 'condition_code' in X.columns:
        X['condition_code'] = X['condition_code'].replace({'良': 0, '稍重': 1, '稍': 1, '重': 2, '不良': 3})

    X = X.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0.0)

    try:
        if hasattr(model, "predict_proba"):
            raw_scores = model.predict_proba(X)[:, 1]
        else:
            raw_scores = model.predict(X)
    except Exception:
        return None

    s = raw_scores.sum()
    win_probs = raw_scores / s if s > 0 else np.ones(len(raw_scores))/len(raw_scores)
    race_df['win_prob'] = win_probs

    std_p = race_df['win_prob'].std()
    if pd.isna(std_p) or std_p < 1e-5:
        race_df['score_brain1'] = 100
    else:
        race_df['score_brain1'] = (100 + (race_df['win_prob'] - race_df['win_prob'].mean()) / std_p * 15).round().astype(int)

    race_df['ev_brain2'] = race_df['win_prob'] * race_df['単勝_num']
    race_df['人気_sort'] = pd.to_numeric(race_df['人気'], errors='coerce').fillna(999)

    return race_df.sort_values(by=['score_brain1', '人気_sort'], ascending=[False, True]).reset_index(drop=True)

# ==========================================
# 4. マーカーとHTMLテーブル生成
# ==========================================
def get_mark(idx, ev, odds, win_prob):
    if idx == 0: return "◎ 本命"
    if idx == 1: return "◯ 対抗"
    if idx == 2: return "▲ 単穴"
    if idx in [3, 4]: return "△ 連下"
    if win_prob <= 0.07: return "消"
    if ev >= 0.7 or odds >= 15.0: return "☆ 穴馬"
    return "消"

def get_all_markers():
    markers = {}
    if df_future.empty: return markers
    for rid in df_future['race_id'].unique():
        sdf = calculate_race_scores(rid, df_future)
        if sdf is not None and len(sdf) >= 5:
            rname = str(sdf['race_name'].iloc[0]) if 'race_name' in sdf.columns else ""
            if "新馬" in rname:
                markers[rid] = "【🐣新馬】"
                continue
            
            top3 = sdf.head(3)
            has_value = False
            for _, row in top3.iterrows():
                pop = pd.to_numeric(row.get('人気', 0), errors='coerce')
                if pd.notna(pop) and pop >= 4:
                    has_value = True
                    break
            
            if has_value:
                markers[rid] = "【🔥妙味】"
            else:
                markers[rid] = "【普通】"
    return markers

markers = get_all_markers()

def generate_beautiful_table(disp_df, is_newcomer):
    html = "<div class='table-container'>"
    html += "<table class='kachi-table'>"
    html += "<thead><tr><th>馬番</th><th style='text-align:left;'>馬名</th><th>騎手</th><th>AIスコア</th><th>勝率</th><th>予想オッズ</th><th>期待値</th><th>印</th></tr></thead>"
    html += "<tbody>"
    
    for i, r in disp_df.iterrows():
        ev_val = float(r.get('ev_brain2', 0))
        raw_o = r.get('単勝') if pd.notna(r.get('単勝')) else r.get('オッズ', 0)
        odds_val = float(pd.to_numeric(raw_o, errors='coerce')) if pd.notna(raw_o) else 0.0
        win_prob_val = float(r.get('win_prob', 0))
        mark = get_mark(i, ev_val, odds_val, win_prob_val)
        
        badge_cls = "badge-keshi"
        if "◎" in mark: badge_cls = "badge-honmei"
        elif "◯" in mark: badge_cls = "badge-taikou"
        elif "▲" in mark: badge_cls = "badge-tana"
        elif "△" in mark: badge_cls = "badge-renka"
        elif "☆" in mark: badge_cls = "badge-ana"
        
        score_str = f"<b>{int(r['score_brain1'])}</b>" if not is_newcomer else "-"
        win_str = f"<b>{win_prob_val*100:.1f}%</b>"
        odds_str = f"{odds_val:.1f}倍" if odds_val > 0 else "-"
        ev_str = f"<b>{ev_val:.2f}</b>" if ev_val > 0 and not is_newcomer else "-"
        
        mark_html = f"<span class='badge-mark {badge_cls}'>{mark}</span>"
        
        html += f"<tr><td style='font-weight:bold; font-size:1.1em; color:#34495e;'>{int(r['馬番']):02d}</td><td style='text-align:left; font-weight:bold; color:#2c3e50;'>{r.get('馬名', '-')}</td><td style='color:#7f8c8d;'>{r.get('騎手', '-')}</td><td style='color:#2c3e50;'>{score_str}</td><td style='color:#2c3e50;'>{win_str}</td><td style='color:#7f8c8d;'>{odds_str}</td><td style='color:#2c3e50;'>{ev_str}</td><td>{mark_html}</td></tr>"
        
    html += "</tbody></table></div>"
    return html

# ==========================================
# 5. メインUI
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
                
                # 【修正】6レースごとに新しく行を作ることで、スマホでも1Rから順番に縦に並ぶようにする
                for i in range(0, len(races), 6):
                    chunk = races[i:i+6]
                    cols = st.columns(6)
                    for j, r in enumerate(chunk):
                        col = cols[j]
                        race_rows = place_df[place_df['r_num'] == r]
                        rid = race_rows['race_id'].iloc[0]
                        rname = str(race_rows['race_name'].iloc[0]).strip() if 'race_name' in race_rows.columns and pd.notna(race_rows['race_name'].iloc[0]) else ""
                        mark = markers.get(rid, "")
                        
                        # ボタンにレース名を追加して表示 (例: 1R 2歳未勝利 【普通】)
                        if rname and rname != "nan":
                            label = f"{r}R {rname} {mark}".strip()
                        else:
                            label = f"{r}R {mark}".strip()
                            
                        btn_type = "primary" if "🔥" in mark else "secondary"
                        if col.button(label, key=f"btn_{rid}", use_container_width=True, type=btn_type):
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
        st.markdown(f"<h2 style='color:#2c3e50;'>🚀 {race_display_name}</h2>", unsafe_allow_html=True)
        
        scored_df = calculate_race_scores(target_id, df_future)
        
        if scored_df is not None:
            st.markdown("<div class='section-header'>📊 1. 出走馬 期待値＆データ一覧</div>", unsafe_allow_html=True)
            disp_df = scored_df.copy().sort_values(by=['score_brain1', 'win_prob'], ascending=[False, False]).reset_index(drop=True)
            if is_newcomer:
                st.info("🐣 新馬戦のため、過去データが存在せずベースAIスコアは参考値です。Geminiの自力予想に委ねます。")
            st.markdown(generate_beautiful_table(disp_df, is_newcomer), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧠 Geminiで最終適正化＆買い目生成", type="primary", use_container_width=True):
            if not GEMINI_API_KEY:
                st.error("【設定エラー】APIキーが見つかりません。")
                st.stop()
            if scored_df is None or len(scored_df) < 6:
                st.error("出走頭数が少ない、またはデータが不足しているため予想をスキップします。")
                st.stop()

            table_summary = []
            for idx, row in disp_df.iterrows():
                ev_val = float(row.get('ev_brain2', 0))
                raw_o = row.get('単勝') if pd.notna(row.get('単勝')) else row.get('オッズ', 0)
                odds_val = float(pd.to_numeric(raw_o, errors='coerce')) if pd.notna(raw_o) else 0.0
                win_prob_val = float(row.get('win_prob', 0))
                mark = get_mark(idx, ev_val, odds_val, win_prob_val)
                table_summary.append(f"馬番:{int(row.get('馬番', 0)):02d} | 馬名:{row.get('馬名', '不明')} | オッズ:{odds_val}倍 ({row.get('人気', 999)}人気) | 純粋スコア:{row.get('score_brain1', 0)} | 期待値:{ev_val:.2f} | システム評価:{mark}")

            system_instruction = f"""
あなたはプロの競馬分析AI「勝ちぱかくん」の最終意思決定者（Gemini脳）です。
【⚠️絶対ルール】検索で全馬を個別に調べるのは禁止です。日付とレース名で一括検索してください。
【出力フォーマット】
---
### 🌪️ レース展開とリアルタイム情報の統合
* （プロの分析）
### 💥 勝ちぱかくんの最終ジャッジ（印と根拠）
* **◎（本命）:** 〇〇番（馬名） - （抜擢理由）
* **◯（対抗）:** 〇〇番（馬名） - （見解）
* **▲（単穴）:** 〇〇番（馬名） - （見解）
* **△（連下）:** 〇〇番、〇〇番
* **☆（穴馬）:** 〇〇番 - （期待値馬など）
### 💡 戦略的・推奨全買い目
* **単勝:** ◎
* **馬連:** ◎ - ◯▲△☆ (流し)
* **ワイド:** ◎ - ◯▲△☆ (流し)
* **三連複:** ◎◯▲△☆の印から選んだ4〜5頭のボックス
* **三連単:** ◎ 1着固定 - ◯▲△☆ (フォーメーション等)
---
"""
            prompt = f"対象レース: {selected_date} {race_display_name}\n\n"
            if is_newcomer: prompt += "【新馬戦】血統や調教を中心に予想を組み立ててください。\n\n"
            prompt += f"出走馬データ:\n{chr(10).join(table_summary)}"

            with st.spinner("AIが本日付の一括検索で調教情報を取得し、全買い目を生成中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                res_text = ""
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash', contents=prompt,
                            config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.3, tools=[{"googleSearch": {}}])
                        )
                        res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                        if res_text: break
                    except: time.sleep(3)
                    
                if res_text:
                    st.markdown(res_text)
                    honmei_match = re.search(r'◎.*?[）:]\s*(\d+)番', res_text)
                    h_umaban = int(honmei_match.group(1)) if honmei_match else int(disp_df.iloc[0]['馬番'])
                    all_nums = re.findall(r'(\d+)番', res_text)
                    partners_str = ",".join(list(dict.fromkeys([n for n in all_nums if int(n) != h_umaban]))[:5])
                    
                    if df_history.empty or str(target_id) not in df_history['race_id'].astype(str).values:
                        new_record = pd.DataFrame([{'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'race_id': str(target_id), 'race_name': race_display_name, 'honmei_umaban': h_umaban, 'partners': partners_str, 'honmei_name': "履歴参照", 'result_pay': "", 'pay_tansho': 0, 'pay_umaren': 0, 'pay_wide': 0, 'pay_sanrenpuku': 0, 'pay_sanrentan_axis': 0, 'pay_sanrentan_form': 0}])
                        df_history = pd.concat([df_history, new_record], ignore_index=True)
                        df_history.to_csv(HISTORY_CSV, index=False, encoding='utf-8-sig')
                    st.success("📝 実戦履歴に記録しました！")
                else:
                    st.warning("⚠️ 回答を取得できませんでした。")

# ==========================================
# 6. ダッシュボード
# ==========================================
with tab_dashboard:
    st.markdown("<div class='section-header'>📈 実戦成績ダッシュボード</div>", unsafe_allow_html=True)
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
            if total_races == 0:
                st.info(f"この期間のレースはありません。")
                return
            
            finished_df = raw_df[pd.to_numeric(raw_df['result_pay'], errors='coerce').notna()]
            pending_df = raw_df[pd.to_numeric(raw_df['result_pay'], errors='coerce').isna()]
            total = len(finished_df)
            
            if total == 0:
                st.markdown(f"**{title_prefix} 確定レース**: 0 件 （結果待ち: {total_races} 件）")
            else:
                hits = len(finished_df[pd.to_numeric(finished_df['result_pay'], errors='coerce') > 0])
                returns = pd.to_numeric(finished_df['result_pay'], errors='coerce').sum()
                
                invested_total = 0
                inv_tansho = inv_umaren = inv_wide = inv_sanrenpuku = inv_sanrentan_axis = inv_sanrentan_form = 0
                
                for _, r in finished_df.iterrows():
                    p_list = [x for x in str(r.get('partners', '')).split(',') if x.strip().isdigit()]
                    p_len = len(p_list)
                    
                    # 【修正】点数計算を「選ばれた相手馬の数」から正確に計算するように統一
                    if p_len > 0:
                        inv_tansho += 100
                        inv_umaren += p_len * 100
                        inv_wide += p_len * 100
                        
                        box_count = p_len + 1
                        if box_count >= 3:
                            inv_sanrenpuku += int(box_count * (box_count - 1) * (box_count - 2) / 6) * 100
                            
                        if p_len >= 2:
                            inv_sanrentan_axis += (p_len * (p_len - 1)) * 100
                            inv_sanrentan_form += (2 * (p_len - 1)) * 100
                
                invested_total = inv_tansho + inv_umaren + inv_wide + inv_sanrenpuku + inv_sanrentan_axis
                roi_total = (returns / invested_total) * 100 if invested_total > 0 else 0.0
                profit_total = int(returns - invested_total)
                
                st.markdown(f"**{title_prefix} 確定レース**: {total} 件 （結果待ち: {len(pending_df)} 件）")
                
                col1, col2, col3 = st.columns(3)
                # 【修正】0/1的中が緑色になってしまう仕様を修正 (delta_color="off")
                col1.metric("🎯 的中率", f"{(hits/total)*100:.1f}%", f"{hits} / {total} レース的中", delta_color="off")
                col2.metric("💰 回収率 (※1着固定ベース)", f"{roi_total:.1f}%", delta_color="normal" if profit_total >= 0 else "inverse")
                col3.metric("💴 収支 (※1着固定ベース)", f"{profit_total:,} 円")
                
                st.markdown("<br><h5>🎫 券種別の詳細データ（3直単 比較検証）</h5>", unsafe_allow_html=True)
                
                def make_ticket_card(col, name, hits_val, returns_val, inv_val):
                    roi_val = (returns_val / inv_val) * 100 if inv_val > 0 else 0
                    profit_val = int(returns_val - inv_val)
                    color = "#e74c3c" if profit_val < 0 else "#2ecc71"
                    sign = "+" if profit_val > 0 else ""
                    col.markdown(f'''
                    <div style="border:1px solid #ddd; padding:10px; border-radius:8px; text-align:center; background-color:#fff; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                        <div style="font-weight:bold; font-size:1.05em; color:#2c3e50; margin-bottom:5px;">{name}</div>
                        <div style="font-size:0.8em; color:#7f8c8d;">投資: {int(inv_val):,}円</div>
                        <div style="font-size:0.8em; color:#7f8c8d;">的中: {(hits_val/total)*100:.1f}%</div>
                        <div style="font-size:0.8em; color:#7f8c8d;">回収: {roi_val:.1f}%</div>
                        <div style="font-weight:bold; color:{color}; margin-top:5px; font-size:1.1em;">{sign}{profit_val:,} 円</div>
                    </div>
                    ''', unsafe_allow_html=True)

                ticket_cols1 = st.columns(3)
                ticket_cols2 = st.columns(3)

                if 'pay_tansho' in finished_df.columns:
                    make_ticket_card(ticket_cols1[0], "単勝", len(finished_df[pd.to_numeric(finished_df['pay_tansho'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_tansho'], errors='coerce').sum(), inv_tansho)
                    make_ticket_card(ticket_cols1[1], "馬連", len(finished_df[pd.to_numeric(finished_df['pay_umaren'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_umaren'], errors='coerce').sum(), inv_umaren)
                    make_ticket_card(ticket_cols1[2], "ワイド", len(finished_df[pd.to_numeric(finished_df['pay_wide'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_wide'], errors='coerce').sum(), inv_wide)
                    make_ticket_card(ticket_cols2[0], "三連複", len(finished_df[pd.to_numeric(finished_df['pay_sanrenpuku'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_sanrenpuku'], errors='coerce').sum(), inv_sanrenpuku)
                    make_ticket_card(ticket_cols2[1], "三連単 (1着固定流し)", len(finished_df[pd.to_numeric(finished_df['pay_sanrentan_axis'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_sanrentan_axis'], errors='coerce').sum(), inv_sanrentan_axis)
                    make_ticket_card(ticket_cols2[2], "三連単 (ﾌｫｰﾒｰｼｮﾝ)", len(finished_df[pd.to_numeric(finished_df['pay_sanrentan_form'], errors='coerce') > 0]), pd.to_numeric(finished_df['pay_sanrentan_form'], errors='coerce').sum(), inv_sanrentan_form)

                st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<h5 style='margin-top:20px;'>📋 レース履歴詳細</h5>", unsafe_allow_html=True)
            
            history_html = ""
            for _, row in raw_df.sort_values(by='date', ascending=False).iterrows():
                date_str = pd.to_datetime(row['date']).strftime('%m/%d %H:%M') if pd.notna(row['date']) else "-"
                pay_val = pd.to_numeric(row.get('result_pay'), errors='coerce')
                
                if pd.isna(pay_val):
                    status_html = "<span style='color:#f39c12; font-weight:bold; font-size:0.9em;'>結果待ち・集計中</span>"
                    border_color = "#f39c12"
                else:
                    p_list = [x for x in str(row.get('partners', '')).split(',') if x.strip().isdigit()]
                    p_len = len(p_list)
                    inv = 0
                    if p_len > 0:
                        inv_tansho_row = 100
                        inv_umaren_row = p_len * 100
                        inv_wide_row = p_len * 100
                        box_count_row = p_len + 1
                        inv_sanrenpuku_row = int(box_count_row * (box_count_row - 1) * (box_count_row - 2) / 6) * 100 if box_count_row >= 3 else 0
                        inv_sanrentan_axis_row = (p_len * (p_len - 1)) * 100 if p_len >= 2 else 0
                        inv = inv_tansho_row + inv_umaren_row + inv_wide_row + inv_sanrenpuku_row + inv_sanrentan_axis_row
                        
                    profit = int(pay_val - inv)
                    
                    if profit > 0:
                        status_html = f"<span style='color:#2ecc71; font-weight:bold; font-size:1.1em;'>+{profit:,}円</span>"
                        border_color = "#2ecc71"
                    else:
                        status_html = f"<span style='color:#e74c3c; font-weight:bold; font-size:1.1em;'>{profit:,}円</span>"
                        border_color = "#e74c3c"

                history_html += f"""
                <div style="background:#fff; border-radius:8px; padding:10px 15px; margin-bottom:8px; border-left:4px solid {border_color}; box-shadow:0 1px 3px rgba(0,0,0,0.05); color:#333;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div style="font-size:0.85em; color:#7f8c8d; margin-bottom:2px;">{date_str} <span style="margin:0 5px;">|</span> {row.get('race_name', '-')}</div>
                            <div style="font-weight:bold; font-size:1.1em; color:#2c3e50;">◎ {row.get('honmei_umaban', '-')}番</div>
                        </div>
                        <div style="text-align:right;">
                            {status_html}
                        </div>
                    </div>
                </div>
                """
            st.markdown(history_html, unsafe_allow_html=True)

        with tab_total:
            render_dashboard_for_df(df_history, "総合")
        with tab_month:
            months = sorted(df_history['year_month'].dropna().unique(), reverse=True)
            if months:
                selected_month = st.selectbox("表示する月を選択", months)
                render_dashboard_for_df(df_history[df_history['year_month'] == selected_month], f"{selected_month} の")
        with tab_day:
            days = sorted(df_history['just_date'].dropna().unique(), reverse=True)
            if days:
                selected_day = st.selectbox("表示する日付を選択", days)
                render_dashboard_for_df(df_history[df_history['just_date'] == selected_day], f"{selected_day} の")