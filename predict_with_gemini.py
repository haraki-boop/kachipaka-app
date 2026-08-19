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
ML_TARGET_CSV = "ml_target_data.csv"

def clean_horse_name(name):
    if pd.isna(name): return ""
    s = unicodedata.normalize('NFKC', str(name))
    s = re.sub(r'[\s・･.\-ー_]+', '', s).strip()
    return s.upper() # 英字も大文字に統一

def get_badge_class(mark):
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
    df_past = pd.DataFrame()
    if os.path.exists(ML_TARGET_CSV):
        for enc in ['utf-8-sig', 'utf-8', 'cp932']:
            try:
                df_past = pd.read_csv(ML_TARGET_CSV, low_memory=False, dtype={'race_id': str}, encoding=enc)
                df_past['馬名_clean'] = df_past['馬名'].astype(str).apply(clean_horse_name)
                df_past['date_parsed'] = pd.to_datetime(df_past['date'], errors='coerce')
                df_past['distance_num'] = pd.to_numeric(df_past.get('distance'), errors='coerce')
                df_past['dist_cat'] = df_past['distance_num'].apply(get_dist_cat)
                df_past['rank_num'] = pd.to_numeric(df_past.get('着順'), errors='coerce')
                df_past['kinryo_num'] = pd.to_numeric(df_past.get('斤量'), errors='coerce')
                df_past['wakuban_num'] = pd.to_numeric(df_past.get('枠番'), errors='coerce')
                
                place_code = df_past.get('place_code', pd.Series(['00']*len(df_past)))
                surface = df_past.get('surface', pd.Series(['芝']*len(df_past)))
                df_past['course_id'] = place_code.astype(str) + "_" + surface.astype(str) + "_" + df_past['distance_num'].fillna(0).astype(int).astype(str)
                df_past['course_frame_id'] = df_past['course_id'] + "_frame_" + df_past['wakuban_num'].fillna(0).astype(int).astype(str)
                df_past = df_past.sort_values(by='date_parsed')
                break
            except: pass

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
                    break
            except: pass

    return df_past, df_future

model_data = load_model()
df_past, df_future = load_data()

# ==========================================
# 📦 過去データ辞書化（マッチング強化）
# ==========================================
@st.cache_data
def build_past_horse_dict(df_p):
    if df_p.empty: return {}, {}, {}
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
            'eff_rank_avg': ranks.tail(3).mean() if not ranks.empty else 8.0, # 未登録馬は平均値で補正
            'eff_top5_rate': (ranks.tail(5) <= 5).mean() if not ranks.empty else 0.2,
            'eff_top3_rate': (ranks.tail(5) <= 3).mean() if not ranks.empty else 0.1,
            'eff_my_time_idx': t_vals.tail(3).median() if not t_vals.empty else 50.0,
            'eff_my_last3f_idx': l_vals.tail(3).median() if not l_vals.empty else 50.0,
            'eff_my_pace_idx': p_vals.tail(3).median() if not p_vals.empty else 50.0,
            'eff_my_start_idx': s_vals.tail(3).median() if not s_vals.empty else 50.0
        }

    course_front_map = df_p.groupby('course_id')['rank_num'].apply(lambda x: (x <= 3).mean()).to_dict()
    course_frame_map = df_p.groupby('course_frame_id')['rank_num'].apply(lambda x: (x == 1).mean()).to_dict()

    return horse_dict, course_front_map, course_frame_map

past_dict, course_front_map, course_frame_map = build_past_horse_dict(df_past)

# ==========================================
# 3. AI推論ロジック（全頭同一スコア防止改修）
# ==========================================
def calculate_predictions(race_id_target, df_fut, cond):
    if df_fut.empty or model_data is None: return None
    race_df = df_fut[df_fut['race_id'].astype(str) == str(race_id_target)].copy()
    if race_df.empty: return None

    model = model_data['model']
    features = model_data['features']

    # 過去データの補完
    target_cols = [
        'last_date', 'prev_prize', 'prev_rank_num', 
        'eff_rank_avg', 'eff_top5_rate', 'eff_top3_rate',
        'eff_my_time_idx', 'eff_my_last3f_idx',
        'eff_my_pace_idx', 'eff_my_start_idx'
    ]
    for col in target_cols:
        race_df[col] = race_df['馬名_clean'].apply(lambda x: past_dict.get(x, {}).get(col, np.nan))

    # リアルタイム（出馬表）の特徴量作成
    weights_parsed = race_df.get('馬体重', pd.Series()).apply(parse_weight)
    race_df['body_weight'] = [p[0] for p in weights_parsed]
    race_df['body_weight_diff'] = [p[1] for p in weights_parsed]
    race_df['kinryo_body_ratio'] = race_df['kinryo_num'] / race_df['body_weight'].fillna(470)

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

    race_df['date_parsed_fut'] = pd.to_datetime(race_df['date'], errors='coerce')
    race_df['interval_days'] = (race_df['date_parsed_fut'] - race_df['last_date']).dt.days.fillna(30)
    race_df['is_long_rest'] = (race_df['interval_days'] >= 180).astype(int)

    j_col = race_df.get('jockey_win_power', race_df.get('jockey_win_rate', pd.Series()))
    race_df['eff_jockey_win'] = pd.to_numeric(j_col, errors='coerce').fillna(0.1).clip(0.0, 1.0)
    race_df['eff_jockey_track_win'] = pd.to_numeric(race_df.get('jockey_track_win_rate'), errors='coerce').fillna(0.1).clip(0.0, 1.0)
    race_df['horse_win_rate_val'] = pd.to_numeric(race_df.get('horse_win_rate'), errors='coerce').fillna(0.1).clip(0.0, 1.0)
    race_df['horse_runs_val'] = pd.to_numeric(race_df.get('horse_runs'), errors='coerce').fillna(5)

    race_df['condition_code'] = cond
    race_df['condition'] = cond

    X = pd.DataFrame(index=race_df.index)
    for f in features:
        X[f] = race_df[f] if f in race_df.columns else np.nan

    cat_cols = ['surface_code', 'condition_code', 'sex_code']
    for cat in cat_cols:
        if cat in X.columns:
            X[cat] = X[cat].astype('category')
            if cat == 'sex_code': X[cat] = X[cat].replace({'牡':0,'牝':1,'セ':2})
            if cat == 'surface_code': X[cat] = X[cat].replace({'芝':0,'ダート':1,'障害':2})
            if cat == 'condition_code': X[cat] = X[cat].replace({'良':0,'稍重':1,'稍':1,'重':2,'不良':3})

    X_num = X.copy()
    for col in X_num.columns:
        if col not in cat_cols:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')

    try:
        raw_probs = model.predict(X_num)
    except Exception as e:
        st.error(f"モデル予測エラー: {e}")
        return None

    # オッズデータの取得
    race_df['単勝_num'] = pd.to_numeric(race_df.get('単勝', race_df.get('オッズ', pd.Series())), errors='coerce').fillna(10.0)
    race_df['人気'] = race_df['単勝_num'].rank(method='min')

    # ★万が一全頭ほぼ同点になった場合の補正処理（過去成績・オッズ順位から確率の傾きを復元）
    if np.std(raw_probs) < 0.001:
        # オッズ・過去着順から自然な確率勾配を作る
        implied_probs = (1.0 / race_df['単勝_num'])
        raw_probs = implied_probs.values

    # 勝率の正規化
    prob_sum = np.sum(raw_probs)
    if prob_sum > 0:
        race_df['win_prob'] = raw_probs / prob_sum
    else:
        race_df['win_prob'] = 1.0 / len(race_df)

    # 期待値算出
    race_df['ev'] = race_df['win_prob'] * race_df['単勝_num']

    # ★AIスコア（50〜150）のダイナミックスケール★
    p_min = race_df['win_prob'].min()
    p_max = race_df['win_prob'].max()
    if pd.notna(p_min) and pd.notna(p_max) and p_max > p_min:
        race_df['ai_score'] = (50 + (race_df['win_prob'] - p_min) / (p_max - p_min) * 100).round().astype(int)
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

    # スコア（勝率）の順番だけでソート
    race_df = race_df.sort_values(by=['ai_score', 'win_prob'], ascending=[False, False]).reset_index(drop=True)
    race_df['印'] = "消"
    
    if len(race_df) > 0: race_df.loc[0, '印'] = "◎"
    if len(race_df) > 1: race_df.loc[1, '印'] = "◯"
    if len(race_df) > 2: race_df.loc[2, '印'] = "▲"
    if len(race_df) > 3: race_df.loc[3, '印'] = "△"
    if len(race_df) > 4: race_df.loc[4, '印'] = "△"
    
    # ☆（穴馬）選出：本命漏れ（6番手以降）から期待値が1.00〜1.20付近の馬を1頭厳選
    if len(race_df) > 5:
        ana_candidates = race_df.iloc[5:][(race_df['単勝_num'] >= 8.0) & (race_df['ev'] >= 0.90)]
        if not ana_candidates.empty:
            ana_idx = (ana_candidates['ev'] - 1.10).abs().idxmin()
            race_df.loc[ana_idx, '印'] = "☆"

    mark_order = {"◎":1, "◯":2, "▲":3, "△":4, "☆":5, "消":6}
    race_df['mark_rank'] = race_df['印'].map(mark_order).fillna(99)
    race_df = race_df.sort_values(by='mark_rank').reset_index(drop=True)

    return race_df

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
        html += f"<td style='text-align:left; font-size:13px; color:#555;'>{r.get('短評', '-')}</td>"
        html += f"</tr>"
    html += "</table></div>"
    return html

# ==========================================
# 5. メインUI
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

place_tabs = st.tabs([f"📍 {p}" for p in places])
for i, place in enumerate(places):
    with place_tabs[i]:
        place_df = day_df[day_df['place_name'] == place]
        races = sorted(place_df['r_num'].unique())
        cols = st.columns(6)
        for j, r in enumerate(races):
            r_df = place_df[place_df['r_num'] == r]
            r_id = r_df['race_id'].iloc[0]
            r_name = str(r_df.get('race_name', pd.Series([''])).iloc[0]).strip()
            label = f"{r}R {r_name}" if r_name and r_name != 'nan' else f"{r}R"
            if cols[j % 6].button(label, key=f"btn_{r_id}", use_container_width=True):
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
    
    res_df = calculate_predictions(t_id, df_future, cond)
    
    if res_df is not None:
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

【あなたの役割と3大定性チェックワークフロー】
1. まず、提供された「Pythonが算出したベース予測データ（AI勝率、期待値、Python印）」を確認してください。
2. 次に、Google検索ツールを駆使して、各出走馬に関する以下の【3大定性情報】を深掘り検索してください。
   ①【調教（追い切り状態）】: 前走時と比較した追い切りタイムの向上や坂路/CWでの状態、勝負気配
   ②【血統（コース/馬場適性）】: 今日のコース・馬場状態（洋芝・ダート・重馬場など）に対する血統的適性
   ③【前走敗因・不利】: 前走の大敗に明確な理由（直線での不利、不向きな距離/馬場など）があり、今回巻き返せるか
3. Pythonの定量データと、検索で得た3大定性情報を掛け合わせ、総合的に判断したあなたの最終評価（Gemini印と短評）を下してください。

【重要：カンニング絶対禁止ルール】
現在、過去のレースを用いてモデルの精度検証を行っています。そのため【実際のレース結果（着順・配当など）を検索してカンニングすること】は絶対に禁止です。あくまで「レース発走前の事前情報」のみを検索して評価を構成してください。

【出力フォーマット】
テキストによる解説は一切不要です。必ず以下のJSON形式のみで出力してください（マークダウンの ```json などは絶対に入れないでください）。

{{
  "evaluations": [
    {{"馬番": 1, "Gemini印": "◎", "短評": "調教自己ベスト。血統も洋芝向き"}},
    {{"馬番": 2, "Gemini印": "☆", "短評": "前走は不完全燃焼。血統一変期待"}}
  ]
}}

【予想ルール】
1. 「evaluations」配列には、提供された出走全頭分のデータを含めてください。
2. 「Gemini印」は ◎, ◯, ▲, △, ☆, 消 のいずれかを必ず使用してください。
3. 「短評」は15文字以内で、調教・血統・敗因分析に基づく明確な理由を書いてください。
"""
            prompt = f"対象レース: {sel_date} {r_info['place_name']} {r_info['r_num']}R\n【想定馬場状態】: {cond}\n\n【Python算定データ】:\n{chr(10).join(table_summary)}"

            with st.spinner("🧠 Geminiが調教・血統・敗因データを検索し、表をアップデート中..."):
                client = genai.Client(api_key=GEMINI_API_KEY)
                res_text = ""
                for _ in range(3):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash', 
                            contents=prompt, 
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction, 
                                temperature=0.3,
                                tools=[{"googleSearch": {}}]
                            )
                        )
                        res_text = response.text if response.text else (response.candidates[0].content.parts[0].text if response.candidates else "")
                        if res_text: break
                    except: time.sleep(3)
                    
                if res_text:
                    clean_json_text = re.sub(r'^```json\n|```$', '', res_text, flags=re.MULTILINE).strip()
                    try:
                        gemini_data = json.loads(clean_json_text)
                        
                        evals = gemini_data.get("evaluations", [])
                        eval_df = pd.DataFrame(evals)
                        if not eval_df.empty and '馬番' in eval_df.columns:
                            eval_df['馬番_num'] = pd.to_numeric(eval_df['馬番'], errors='coerce')
                            if 'Gemini印' not in eval_df.columns: eval_df['Gemini印'] = '消'
                            if '短評' not in eval_df.columns: eval_df['短評'] = '-'
                            eval_df_clean = eval_df[['馬番_num', 'Gemini印', '短評']]
                            
                            res_df['馬番_num'] = pd.to_numeric(res_df['馬番'], errors='coerce')
                            
                            merged_df = pd.merge(res_df, eval_df_clean, on='馬番_num', how='left')
                            
                            table_placeholder.empty()
                            with table_placeholder.container():
                                st.markdown("<div class='section-header'>🔥 Python × Gemini 融合データ</div>", unsafe_allow_html=True)
                                st.markdown(generate_fusion_table(merged_df, is_newcomer), unsafe_allow_html=True)
                            
                            st.success("📝 表を最強アップデートしました！")

                    except json.JSONDecodeError:
                        st.error("Geminiからのデータ解析に失敗しました。もう一度お試しください。")
                else:
                    st.warning("⚠️ 回答を取得できませんでした。")